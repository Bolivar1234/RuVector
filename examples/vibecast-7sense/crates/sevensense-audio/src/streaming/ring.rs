//! Lock-free single-producer single-consumer ring buffer for live audio.
//!
//! The producer is a real-time audio callback: it must never allocate, lock, or
//! block. When the consumer falls behind, the producer overwrites the oldest
//! samples and the consumer detects the lapse from the position counters. ADR-010
//! records why lossy overwrite is preferred to back-pressure here -- stalling an
//! audio callback causes device-level glitches, and in a monitoring deployment
//! stale audio is worth less than current audio.
//!
//! Samples are stored as [`AtomicU32`] bit patterns rather than plain `f32`. A
//! torn read is possible by construction under an overwrite policy, and going
//! through atomics makes that a stale value rather than undefined behaviour.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// A bounded SPSC ring buffer that overwrites its oldest samples when full.
///
/// Capacity is rounded up to a power of two so index wrapping is a mask.
///
/// # Example
/// ```
/// use sevensense_audio::streaming::RingBuffer;
///
/// let ring = RingBuffer::new(8);
/// ring.write(&[1.0, 2.0, 3.0]);
///
/// let mut out = [0.0f32; 3];
/// let (read, dropped) = ring.read(&mut out);
/// assert_eq!(read, 3);
/// assert_eq!(dropped, 0);
/// assert_eq!(out, [1.0, 2.0, 3.0]);
/// ```
#[derive(Debug)]
pub struct RingBuffer {
    slots: Box<[AtomicU32]>,
    mask: usize,
    /// Total samples ever written. Only the producer stores to this.
    write_pos: AtomicU64,
    /// Total samples ever consumed. Only the consumer stores to this.
    read_pos: AtomicU64,
    /// Total samples lost to overwrite, accumulated by the consumer.
    dropped: AtomicU64,
}

impl RingBuffer {
    /// Creates a ring buffer holding at least `capacity` samples.
    ///
    /// # Panics
    /// Panics if `capacity` is zero.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "ring buffer capacity must be non-zero");
        let capacity = capacity.next_power_of_two();
        let slots = (0..capacity)
            .map(|_| AtomicU32::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            slots,
            mask: capacity - 1,
            write_pos: AtomicU64::new(0),
            read_pos: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        }
    }

    /// Creates a ring buffer sized to hold `seconds` of audio at `sample_rate`.
    #[must_use]
    pub fn with_duration(seconds: f32, sample_rate: u32) -> Self {
        let samples = (seconds * sample_rate as f32).ceil() as usize;
        Self::new(samples.max(1))
    }

    /// Number of samples the buffer can hold.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Samples currently available to read, saturating at capacity.
    #[must_use]
    pub fn available(&self) -> usize {
        let write = self.write_pos.load(Ordering::Acquire);
        let read = self.read_pos.load(Ordering::Relaxed);
        ((write - read) as usize).min(self.capacity())
    }

    /// Total samples lost to overwrite since creation.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Total samples ever written.
    #[must_use]
    pub fn written(&self) -> u64 {
        self.write_pos.load(Ordering::Acquire)
    }

    /// Writes samples, overwriting the oldest when full. Never blocks.
    ///
    /// This is the only method the producer may call. It performs no allocation,
    /// so it is safe to invoke from a real-time audio callback.
    pub fn write(&self, samples: &[f32]) {
        let mut pos = self.write_pos.load(Ordering::Relaxed);
        for &sample in samples {
            self.slots[(pos as usize) & self.mask].store(sample.to_bits(), Ordering::Relaxed);
            pos += 1;
        }
        // Release so a consumer that observes this position also observes the
        // sample stores above.
        self.write_pos.store(pos, Ordering::Release);
    }

    /// Reads up to `out.len()` samples.
    ///
    /// Returns `(samples_read, samples_dropped)`, where `samples_dropped` counts
    /// samples overwritten before this read could reach them. A non-zero drop
    /// count means the consumer was lapped and the stream has a gap.
    ///
    /// This is the only method the consumer may call.
    pub fn read(&self, out: &mut [f32]) -> (usize, u64) {
        let write = self.write_pos.load(Ordering::Acquire);
        let mut read = self.read_pos.load(Ordering::Relaxed);
        let capacity = self.capacity() as u64;

        // If the producer has lapped us, skip forward to the oldest sample that
        // still exists. Reporting the gap is the point: a silent skip would look
        // like clean audio with a discontinuity in it.
        let mut dropped = 0;
        if write.saturating_sub(read) > capacity {
            dropped = write - read - capacity;
            read = write - capacity;
            self.dropped.fetch_add(dropped, Ordering::Relaxed);
        }

        let available = (write - read) as usize;
        let count = available.min(out.len());
        for (i, slot) in out.iter_mut().enumerate().take(count) {
            let bits = self.slots[((read + i as u64) as usize) & self.mask].load(Ordering::Relaxed);
            *slot = f32::from_bits(bits);
        }

        self.read_pos.store(read + count as u64, Ordering::Release);
        (count, dropped)
    }

    /// Discards all buffered samples without reading them.
    pub fn clear(&self) {
        let write = self.write_pos.load(Ordering::Acquire);
        self.read_pos.store(write, Ordering::Release);
    }
}

// `Send` and `Sync` are derived automatically: every field is an atomic, so no
// `unsafe impl` is needed. Correctness under concurrent use rests on the SPSC
// contract documented on `read` and `write`, not on a manual marker.
const _: fn() = || {
    fn assert_shareable<T: Send + Sync>() {}
    assert_shareable::<RingBuffer>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_rounds_up_to_a_power_of_two() {
        assert_eq!(RingBuffer::new(3).capacity(), 4);
        assert_eq!(RingBuffer::new(1000).capacity(), 1024);
        assert_eq!(RingBuffer::new(1024).capacity(), 1024);
    }

    #[test]
    fn duration_constructor_sizes_for_the_sample_rate() {
        let ring = RingBuffer::with_duration(1.0, 32_000);
        assert!(ring.capacity() >= 32_000);
    }

    #[test]
    fn write_then_read_round_trips() {
        let ring = RingBuffer::new(16);
        ring.write(&[1.0, -2.5, 3.25]);

        let mut out = [0.0f32; 4];
        let (count, dropped) = ring.read(&mut out);

        assert_eq!(count, 3);
        assert_eq!(dropped, 0);
        assert_eq!(&out[..3], &[1.0, -2.5, 3.25]);
    }

    #[test]
    fn reading_an_empty_buffer_yields_nothing() {
        let ring = RingBuffer::new(8);
        let mut out = [0.0f32; 4];
        assert_eq!(ring.read(&mut out), (0, 0));
    }

    #[test]
    fn overwrite_reports_the_gap_and_keeps_the_newest_samples() {
        let ring = RingBuffer::new(4);
        // Write ten samples into a four-slot buffer: six are lost.
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        ring.write(&input);

        let mut out = [0.0f32; 4];
        let (count, dropped) = ring.read(&mut out);

        assert_eq!(count, 4);
        assert_eq!(dropped, 6, "six samples were overwritten before being read");
        assert_eq!(out, [6.0, 7.0, 8.0, 9.0], "the newest samples survive");
        assert_eq!(ring.dropped(), 6);
    }

    #[test]
    fn available_saturates_at_capacity() {
        let ring = RingBuffer::new(4);
        ring.write(&[0.0; 100]);
        assert_eq!(ring.available(), 4);
    }

    #[test]
    fn interleaved_writes_and_reads_preserve_order() {
        let ring = RingBuffer::new(8);
        let mut out = [0.0f32; 2];
        let mut received = Vec::new();

        for chunk in 0..10 {
            ring.write(&[chunk as f32 * 2.0, chunk as f32 * 2.0 + 1.0]);
            let (count, dropped) = ring.read(&mut out);
            assert_eq!(dropped, 0, "consumer keeps up, so nothing should drop");
            received.extend_from_slice(&out[..count]);
        }

        let expected: Vec<f32> = (0..20).map(|i| i as f32).collect();
        assert_eq!(received, expected);
    }

    #[test]
    fn clear_discards_pending_samples() {
        let ring = RingBuffer::new(8);
        ring.write(&[1.0, 2.0, 3.0]);
        ring.clear();

        let mut out = [0.0f32; 4];
        assert_eq!(ring.read(&mut out).0, 0);
    }

    #[test]
    fn survives_a_concurrent_producer_and_consumer() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let ring = Arc::new(RingBuffer::new(1024));
        let done = Arc::new(AtomicBool::new(false));
        const TOTAL: usize = 100_000;

        let producer = {
            let ring = Arc::clone(&ring);
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                for i in 0..TOTAL {
                    ring.write(&[i as f32]);
                }
                done.store(true, Ordering::Release);
            })
        };

        let consumer = {
            let ring = Arc::clone(&ring);
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                let mut out = vec![0.0f32; 256];
                let mut seen = 0u64;
                let mut lost = 0u64;
                loop {
                    let (count, dropped) = ring.read(&mut out);
                    seen += count as u64;
                    lost += dropped;
                    if count == 0 && done.load(Ordering::Acquire) && ring.available() == 0 {
                        break;
                    }
                }
                (seen, lost)
            })
        };

        producer.join().unwrap();
        let (seen, lost) = consumer.join().unwrap();

        // Every sample is either consumed or accounted for as dropped. Nothing
        // may vanish silently, and nothing may be counted twice.
        assert_eq!(
            seen + lost,
            TOTAL as u64,
            "consumed {seen} + dropped {lost} should equal {TOTAL}"
        );
    }
}
