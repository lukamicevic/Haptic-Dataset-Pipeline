use hound::{WavReader, WavWriter, WavSpec};
use std::fs;
use std::io::{BufWriter, Cursor};
use std::path::Path;

pub struct AudioData {
    pub samples: Vec<i16>,
    pub sample_rate: u32,
}

pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<AudioData, Box<dyn std::error::Error>> {
    // Read entire file in ONE syscall
    let bytes = fs::read(path.as_ref())?;

    // Parse from memory - no more disk I/O
    let cursor = Cursor::new(bytes);
    let mut reader = WavReader::new(cursor)?;

    let samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()?;
    let sample_rate = reader.spec().sample_rate;

    Ok(AudioData { samples, sample_rate })
}

pub fn save_wav<P: AsRef<Path>>(
    path: P,
    samples: &[i16],
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    // BufWriter batches small writes into 8KB chunks
    let file = fs::File::create(path.as_ref())?;
    let buf_writer = BufWriter::new(file);
    let mut writer = WavWriter::new(buf_writer, spec)?;

    // Bulk writer - no Result check per sample
    let mut i16_writer = writer.get_i16_writer(samples.len() as u32);
    for &sample in samples {
        i16_writer.write_sample(sample);
    }
    i16_writer.flush()?;

    Ok(())
}
