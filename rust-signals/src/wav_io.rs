use hound::{WavReader, WavWriter, WavSpec};
use std::path::Path;

pub struct AudioData {
    pub samples: Vec<i16>,
    pub sample_rate: u32,
}

pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<AudioData, Box<dyn std::error::Error>> {
    let mut reader = WavReader::open(path)?;
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

    let mut writer = WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;

    Ok(())
}
