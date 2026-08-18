/**
 * Length-prefixed binary framing for the env <-> Python TCP protocol.
 *
 * Frame layout on the wire:
 *   [4B LE uint32 totalLen][4B LE uint32 headerLen][headerJson][blobs...]
 * where totalLen counts everything after itself.
 *
 * The JSON header carries control fields plus a `tensors` table describing
 * the binary blobs (dtype, shape, byte offset/length within the blob region).
 * This avoids any serialization library and maps directly onto numpy
 * frombuffer on the Python side.
 */

export type DType = "f32" | "u8" | "i32";

export interface TensorSpec {
  dtype: DType;
  shape: number[];
  offset: number; // byte offset within the blob region
  length: number; // byte length
}

export interface Frame {
  header: Record<string, unknown>;
  blobs: Buffer[];
}

export function dtypeSize(dtype: DType): number {
  switch (dtype) {
    case "f32":
    case "i32":
      return 4;
    case "u8":
      return 1;
  }
}

/** Encode a header plus typed-array blobs into one wire frame. */
export function encodeFrame(
  header: Record<string, unknown>,
  tensors: Record<string, { dtype: DType; data: Float32Array | Uint8Array | Int32Array }> = {},
): Buffer {
  const tensorSpecs: Record<string, TensorSpec> = {};
  const blobs: Buffer[] = [];
  let offset = 0;
  for (const [name, t] of Object.entries(tensors)) {
    const buf = Buffer.from(t.data.buffer, t.data.byteOffset, t.data.byteLength);
    tensorSpecs[name] = {
      dtype: t.dtype,
      shape: [t.data.length],
      offset,
      length: buf.length,
    };
    blobs.push(buf);
    offset += buf.length;
  }
  const headerBuf = Buffer.from(
    JSON.stringify({ ...header, tensors: tensorSpecs }),
    "utf8",
  );
  const out = Buffer.alloc(8 + headerBuf.length + offset);
  out.writeUInt32LE(4 + headerBuf.length + offset, 0);
  out.writeUInt32LE(headerBuf.length, 4);
  headerBuf.copy(out, 8);
  let pos = 8 + headerBuf.length;
  for (const b of blobs) {
    b.copy(out, pos);
    pos += b.length;
  }
  return out;
}

/** Incrementally decodes frames from a stream. Feed chunks, get frames. */
export class FrameDecoder {
  private buf: Buffer = Buffer.alloc(0);

  push(chunk: Buffer): Frame[] {
    this.buf = this.buf.length === 0 ? chunk : Buffer.concat([this.buf, chunk]);
    const frames: Frame[] = [];
    for (;;) {
      if (this.buf.length < 4) break;
      const totalLen = this.buf.readUInt32LE(0);
      if (this.buf.length < 4 + totalLen) break;
      const headerLen = this.buf.readUInt32LE(4);
      const header = JSON.parse(
        this.buf.subarray(8, 8 + headerLen).toString("utf8"),
      ) as Record<string, unknown>;
      const blobStart = 8 + headerLen;
      const blobLen = totalLen - 4 - headerLen;
      const blob = this.buf.subarray(blobStart, blobStart + blobLen);
      const tensors = (header.tensors ?? {}) as Record<string, TensorSpec>;
      const blobs: Buffer[] = [];
      const ordered = Object.entries(tensors).sort(
        (a, b) => a[1].offset - b[1].offset,
      );
      for (const [, spec] of ordered) {
        blobs.push(blob.subarray(spec.offset, spec.offset + spec.length));
      }
      frames.push({ header, blobs });
      this.buf = this.buf.subarray(4 + totalLen);
    }
    return frames;
  }
}

/** Convenience: decode a frame into named raw buffers keyed by tensor name. */
export function frameTensors(frame: Frame): Record<string, { spec: TensorSpec; buf: Buffer }> {
  const tensors = (frame.header.tensors ?? {}) as Record<string, TensorSpec>;
  const ordered = Object.entries(tensors).sort(
    (a, b) => a[1].offset - b[1].offset,
  );
  const out: Record<string, { spec: TensorSpec; buf: Buffer }> = {};
  // FrameDecoder already stores offset-ordered subarrays; skip a concat copy.
  if (ordered.length === frame.blobs.length) {
    for (let i = 0; i < ordered.length; i++) {
      const [name, spec] = ordered[i];
      out[name] = { spec, buf: frame.blobs[i] };
    }
    return out;
  }
  const blobRegion = Buffer.concat(frame.blobs);
  for (const [name, spec] of Object.entries(tensors)) {
    out[name] = {
      spec,
      buf: blobRegion.subarray(spec.offset, spec.offset + spec.length),
    };
  }
  return out;
}
