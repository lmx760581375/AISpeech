class MicCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || !input.length || !input[0] || !input[0].length) {
      return true;
    }
    const channel = input[0];
    const copy = new Float32Array(channel.length);
    copy.set(channel);
    this.port.postMessage({ samples: copy.buffer }, [copy.buffer]);
    return true;
  }
}

registerProcessor("mic-capture-processor", MicCaptureProcessor);
