class AudioStreamPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.currentChunk = null;
    this.currentOffset = 0;
    this.started = false;
    this.underrunReported = false;
    this.bufferedFrames = 0;

    this.port.onmessage = (event) => {
      const data = event.data || {};
      if (data.type === "reset") {
        this.queue = [];
        this.currentChunk = null;
        this.currentOffset = 0;
        this.started = false;
        this.underrunReported = false;
        this.bufferedFrames = 0;
        return;
      }

      if (data.type === "append" && data.samples) {
        const chunk = new Float32Array(data.samples);
        if (chunk.length) {
          this.queue.push(chunk);
          this.bufferedFrames += chunk.length;
          this.underrunReported = false;
        }
      }
    };
  }

  process(inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);

    let writeOffset = 0;
    while (writeOffset < output.length) {
      if (!this.currentChunk || this.currentOffset >= this.currentChunk.length) {
        if (!this.queue.length) {
          if (this.started && !this.underrunReported) {
            this.port.postMessage({ type: "underrun" });
            this.underrunReported = true;
          }
          break;
        }
        this.currentChunk = this.queue.shift();
        this.currentOffset = 0;
      }

      const available = this.currentChunk.length - this.currentOffset;
      const required = output.length - writeOffset;
      const count = Math.min(available, required);
      output.set(this.currentChunk.subarray(this.currentOffset, this.currentOffset + count), writeOffset);
      this.currentOffset += count;
      writeOffset += count;
      this.bufferedFrames -= count;
    }

    if (writeOffset > 0 && !this.started) {
      this.started = true;
      this.port.postMessage({ type: "started" });
    }

    return true;
  }
}

registerProcessor("audio-stream-player", AudioStreamPlayerProcessor);
