"""Tests for executor abstraction (local and worker)."""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.models.auth import User
from aegra_api.models.run_job import RunExecution, RunIdentity, RunJob
from aegra_api.services.local_executor import LocalExecutor
from aegra_api.services.run_executor import _timeout_cancellations


async def _empty_async_gen() -> AsyncGenerator[None, None]:
    return
    yield  # noqa: RET504 — makes this an async generator


async def _hanging_async_gen() -> AsyncGenerator[None, None]:
    await asyncio.sleep(9999)
    yield  # noqa: RET504 — makes this an async generator


def _make_job(run_id: str = "run-1") -> RunJob:
    return RunJob(
        identity=RunIdentity(run_id=run_id, thread_id="thread-1", graph_id="graph-1"),
        user=User(identity="user-1"),
        execution=RunExecution(input_data={"msg": "hello"}),
    )


class TestLocalExecutor:
    @pytest.mark.asyncio
    async def test_submit_creates_task(self) -> None:
        executor = LocalExecutor()
        mock_execute = AsyncMock()

        with (
            patch("aegra_api.services.run_executor.execute_run", mock_execute),
            patch("aegra_api.services.local_executor.make_run_trace_context", return_value=None),
        ):
            job = _make_job()
            await executor.submit(job)

            # Task should be registered in active_runs
            from aegra_api.core.active_runs import active_runs

            assert "run-1" in active_runs
            task = active_runs.pop("run-1")
            task.cancel()

    @pytest.mark.asyncio
    async def test_wait_for_completion_returns_on_done(self) -> None:
        executor = LocalExecutor()

        # Create a task that completes immediately
        async def quick() -> None:
            pass

        from aegra_api.core.active_runs import active_runs

        task = asyncio.create_task(quick())
        active_runs["run-done"] = task
        await asyncio.sleep(0.01)

        await executor.wait_for_completion("run-done", timeout=1.0)
        active_runs.pop("run-done", None)

    @pytest.mark.asyncio
    async def test_wait_for_completion_returns_on_missing_run(self) -> None:
        executor = LocalExecutor()
        # Should return immediately, not raise
        await executor.wait_for_completion("nonexistent", timeout=1.0)

    @pytest.mark.asyncio
    async def test_stop_cancels_active_tasks(self) -> None:
        executor = LocalExecutor()

        from aegra_api.core.active_runs import active_runs

        async def hang_forever() -> None:
            await asyncio.sleep(9999)

        task = asyncio.create_task(hang_forever())
        active_runs["run-hang"] = task

        await executor.stop()
        # Give event loop a tick to process cancellation
        await asyncio.sleep(0.01)
        assert task.done()
        active_runs.pop("run-hang", None)

    @pytest.mark.asyncio
    async def test_submit_times_out_and_marks_run_error(self) -> None:
        executor = LocalExecutor()
        timeout_error = "Job exceeded maximum execution time"

        async def slow_execute(_job: RunJob) -> None:
            await asyncio.sleep(9999)

        with (
            patch("aegra_api.services.run_executor.execute_run", side_effect=slow_execute),
            patch("aegra_api.services.local_executor.make_run_trace_context", return_value=None),
            patch("aegra_api.services.local_executor.settings", create=True) as mock_settings,
            patch(
                "aegra_api.services.local_executor.finalize_run", new_callable=AsyncMock, create=True
            ) as mock_finalize,
        ):
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 0.01

            job = _make_job()
            await executor.submit(job)

            from aegra_api.core.active_runs import active_runs

            task = active_runs["run-1"]
            await asyncio.sleep(0.05)
            assert task.done()

        mock_finalize.assert_awaited_once_with(
            "run-1",
            "thread-1",
            status="error",
            thread_status="error",
            error=timeout_error,
        )
        assert "run-1" not in active_runs

    @pytest.mark.asyncio
    async def test_timeout_produces_single_error_finalize_not_double(self) -> None:
        """Timeout must call finalize_run exactly once with status='error'.

        Without the shield fix, asyncio.wait_for cancels execute_run directly.
        execute_run's CancelledError handler calls finalize_run('interrupted'),
        then the TimeoutError handler calls finalize_run('error') — two DB writes.
        """
        executor = LocalExecutor()
        finalize_calls: list[str] = []

        async def realistic_execute(job: RunJob) -> None:
            try:
                await asyncio.sleep(9999)
            except asyncio.CancelledError:
                if job.identity.run_id not in _timeout_cancellations:
                    finalize_calls.append("interrupted")
                raise

        async def track_finalize(run_id: str, thread_id: str, *, status: str, **kwargs: object) -> None:
            finalize_calls.append(status)

        with (
            patch("aegra_api.services.run_executor.execute_run", side_effect=realistic_execute),
            patch("aegra_api.services.local_executor.make_run_trace_context", return_value=None),
            patch("aegra_api.services.local_executor.settings", create=True) as mock_settings,
            patch("aegra_api.services.local_executor.finalize_run", side_effect=track_finalize, create=True),
        ):
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 0.01

            job = _make_job()
            await executor.submit(job)

            from aegra_api.core.active_runs import active_runs

            task = active_runs["run-1"]
            await asyncio.sleep(0.05)
            assert task.done()

        assert finalize_calls == ["error"], f"Expected single error finalize, got: {finalize_calls}"

    @pytest.mark.asyncio
    async def test_timeout_cancellations_flag_causes_execute_run_to_skip_interrupted_finalize(self) -> None:
        """execute_run skips finalize_run when run_id is in _timeout_cancellations."""
        run_id = "run-flag-test"
        _timeout_cancellations.add(run_id)

        finalize_calls: list[str] = []

        async def track_finalize(*args: object, status: str, **kwargs: object) -> None:
            finalize_calls.append(status)

        try:
            with (
                patch("aegra_api.services.run_executor.update_run_status", new_callable=AsyncMock),
                patch("aegra_api.services.run_executor.finalize_run", side_effect=track_finalize),
                patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
                patch("aegra_api.services.run_executor.stream_graph_events", return_value=_hanging_async_gen()),
                patch("aegra_api.services.run_executor.get_langgraph_service") as mock_svc,
                patch("aegra_api.services.run_executor._signal_run_done", new_callable=AsyncMock),
                patch("aegra_api.services.run_executor.active_runs", {}),
            ):
                mock_streaming.cleanup_run = AsyncMock()
                mock_streaming.signal_run_cancelled = AsyncMock()
                mock_streaming.signal_run_error = AsyncMock()

                mock_graph = MagicMock()
                mock_graph.__aenter__ = AsyncMock(return_value=mock_graph)
                mock_graph.__aexit__ = AsyncMock(return_value=False)
                mock_svc.return_value.get_graph = MagicMock(return_value=mock_graph)
                mock_svc.return_value.list_graphs = MagicMock(return_value=["g1"])

                from aegra_api.services.run_executor import execute_run

                task = asyncio.create_task(execute_run(_make_job(run_id)))
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

                assert finalize_calls == [], (
                    f"finalize_run should not be called when run_id is in _timeout_cancellations, got: {finalize_calls}"
                )
                mock_streaming.signal_run_cancelled.assert_not_awaited()
                mock_streaming.signal_run_error.assert_awaited_once()
        finally:
            _timeout_cancellations.discard(run_id)

    @pytest.mark.asyncio
    async def test_submit_cancel_does_not_mark_timeout_error(self) -> None:
        executor = LocalExecutor()

        async def slow_execute(_job: RunJob) -> None:
            await asyncio.sleep(9999)

        with (
            patch("aegra_api.services.run_executor.execute_run", side_effect=slow_execute),
            patch("aegra_api.services.local_executor.make_run_trace_context", return_value=None),
            patch("aegra_api.services.local_executor.settings", create=True) as mock_settings,
            patch(
                "aegra_api.services.local_executor.finalize_run", new_callable=AsyncMock, create=True
            ) as mock_finalize,
        ):
            mock_settings.worker.BG_JOB_TIMEOUT_SECS = 3600

            job = _make_job()
            await executor.submit(job)

            from aegra_api.core.active_runs import active_runs

            task = active_runs["run-1"]
            await asyncio.sleep(0)
            task.cancel()
            results = await asyncio.gather(task, return_exceptions=True)

        assert len(results) == 1
        assert isinstance(results[0], asyncio.CancelledError)
        mock_finalize.assert_not_awaited()
        assert "run-1" not in active_runs

    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout_does_not_cancel_task(self) -> None:
        executor = LocalExecutor()

        from aegra_api.core.active_runs import active_runs

        async def hang_forever() -> None:
            await asyncio.sleep(9999)

        task = asyncio.create_task(hang_forever())
        active_runs["run-wait"] = task

        await executor.wait_for_completion("run-wait", timeout=0.01)

        assert not task.done()

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        active_runs.pop("run-wait", None)


class TestRunExecutorBoundaryConditions:
    """Boundary condition tests for run_executor edge cases."""

    @pytest.mark.asyncio
    async def test_empty_context_passed_as_dict_not_none(self) -> None:
        """Empty context {} must reach get_graph as {}, not None.

        Regression test: `context or None` evaluates to None because
        empty dict is falsy. Factory graphs that read context.model
        crash with AttributeError if context is None.
        """
        job = RunJob(
            identity=RunIdentity(run_id="r1", thread_id="t1", graph_id="g1"),
            user=User(identity="u1"),
            execution=RunExecution(context={}),
        )

        mock_graph = MagicMock()
        mock_graph.__aenter__ = AsyncMock(return_value=mock_graph)
        mock_graph.__aexit__ = AsyncMock(return_value=False)

        mock_service = MagicMock()
        mock_service.get_graph = MagicMock(return_value=mock_graph)

        with (
            patch("aegra_api.services.run_executor.get_langgraph_service", return_value=mock_service),
            patch("aegra_api.services.run_executor.update_run_status", new_callable=AsyncMock),
            patch("aegra_api.services.run_executor.finalize_run", new_callable=AsyncMock),
            patch("aegra_api.services.run_executor.streaming_service") as mock_streaming,
            patch("aegra_api.services.run_executor.stream_graph_events", return_value=_empty_async_gen()),
        ):
            mock_streaming.cleanup_run = AsyncMock()
            mock_streaming.signal_run_error = AsyncMock()

            from aegra_api.services.run_executor import execute_run

            await execute_run(job)

            # Verify context was passed as {} not None
            call_kwargs = mock_service.get_graph.call_args
            assert call_kwargs.kwargs["context"] == {}, (
                f"Expected context={{}}, got context={call_kwargs.kwargs['context']}"
            )


class TestExecutorFactory:
    def test_creates_local_when_redis_disabled(self) -> None:
        with patch("aegra_api.services.executor.settings") as mock_settings:
            mock_settings.redis.REDIS_BROKER_ENABLED = False
            from aegra_api.services.executor import _create_executor

            result = _create_executor()
            assert isinstance(result, LocalExecutor)

    def test_creates_worker_when_redis_enabled(self) -> None:
        with patch("aegra_api.services.executor.settings") as mock_settings:
            mock_settings.redis.REDIS_BROKER_ENABLED = True
            from aegra_api.services.executor import _create_executor
            from aegra_api.services.worker_executor import WorkerExecutor

            result = _create_executor()
            assert isinstance(result, WorkerExecutor)
