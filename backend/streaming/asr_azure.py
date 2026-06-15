"""
Streaming ASR using Azure Speech Services.
"""

import asyncio
import azure.cognitiveservices.speech as speechsdk


class StreamingASR:
    """Azure Speech-to-Text with streaming (partial + final results)."""

    def __init__(self, key: str, region: str):
        self._config = speechsdk.SpeechConfig(subscription=key, region=region)
        self._config.speech_recognition_language = "zh-CN"
        self._config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_EnableAudioLogging, "False"
        )

    async def transcribe_stream(self, audio_chunk_iter):
        """
        Feed audio chunks from an async iterator, yield (type, text) where type is
        "partial" or "final".
        """
        push_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self._config, audio_config=audio_config
        )

        loop = asyncio.get_event_loop()
        result_queue = asyncio.Queue()
        done_event = asyncio.Event()

        def _on_partial(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech and evt.result.text:
                loop.call_soon_threadsafe(result_queue.put_nowait, ("partial", evt.result.text))

        def _on_final(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
                loop.call_soon_threadsafe(result_queue.put_nowait, ("final", evt.result.text))
            loop.call_soon_threadsafe(done_event.set)

        def _on_cancel(evt):
            loop.call_soon_threadsafe(done_event.set)

        recognizer.recognizing.connect(_on_partial)
        recognizer.recognized.connect(_on_final)
        recognizer.canceled.connect(_on_cancel)

        recognizer.start_continuous_recognition()

        async def _feed():
            async for chunk in audio_chunk_iter:
                push_stream.write(chunk)
            push_stream.close()

        feed_task = asyncio.create_task(_feed())
        done_task = asyncio.create_task(self._wait_for(done_event))

        while True:
            get_task = asyncio.create_task(result_queue.get())
            done, pending = await asyncio.wait(
                [get_task, done_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            if done_task in done:
                if not result_queue.empty():
                    yield result_queue.get_nowait()
                break
            yield await get_task

        recognizer.stop_continuous_recognition()
        await feed_task

    async def transcribe_once(self, audio_bytes: bytes) -> str:
        """Non-streaming: transcribe full audio at once."""
        push_stream = speechsdk.audio.PushAudioInputStream()
        push_stream.write(audio_bytes)
        push_stream.close()
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self._config, audio_config=audio_config
        )
        fut = asyncio.get_event_loop().create_future()

        def _on_result(evt):
            if not fut.done():
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    fut.set_result(evt.result.text)
                else:
                    fut.set_result("")

        def _on_cancel(evt):
            if not fut.done():
                fut.set_result("")

        recognizer.recognized.connect(_on_result)
        recognizer.canceled.connect(_on_cancel)
        recognizer.recognize_once()
        return await fut

    @staticmethod
    async def _wait_for(event: asyncio.Event):
        await event.wait()
        return 0.0
