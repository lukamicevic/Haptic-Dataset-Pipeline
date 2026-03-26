use std::fs;
use std::path::Path;

pub struct AudioData {
    pub samples: Vec<i16>,
    pub sample_rate: u32,
}

/// Load WAV file - minimal parsing, zero-copy byte casting
pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<AudioData, Box<dyn std::error::Error + Send + Sync>> {
    let bytes = fs::read(path.as_ref())?;

    // Verify RIFF/WAVE header
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("Not a valid WAV file".into());
    }

    // Read sample rate from fmt chunk (offset 24)
    let sample_rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);

    // Find "data" chunk
    let data_start = find_data_chunk(&bytes)?;
    let data_size = u32::from_le_bytes([
        bytes[data_start + 4],
        bytes[data_start + 5],
        bytes[data_start + 6],
        bytes[data_start + 7],
    ]) as usize;

    let audio_start = data_start + 8;
    let audio_end = (audio_start + data_size).min(bytes.len());
    let audio_bytes = &bytes[audio_start..audio_end];

    // Zero-copy cast: &[u8] -> &[i16] -> Vec<i16>
    let samples: Vec<i16> = bytemuck::cast_slice(audio_bytes).to_vec();

    Ok(AudioData { samples, sample_rate })
}

fn find_data_chunk(bytes: &[u8]) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let mut pos = 12; // After "RIFF" + size + "WAVE"
    while pos + 8 < bytes.len() {
        if &bytes[pos..pos + 4] == b"data" {
            return Ok(pos);
        }
        let chunk_size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        pos += 8 + chunk_size;
    }
    Err("No data chunk found".into())
}

/// Save WAV file - writes header + samples in one syscall
pub fn save_wav<P: AsRef<Path>>(
    path: P,
    samples: &[i16],
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * u32::from(num_channels) * u32::from(bits_per_sample) / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;

    // Pre-allocate exact size: 44 byte header + data
    let mut bytes = Vec::with_capacity(44 + data_size as usize);

    // RIFF header
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&file_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVE");

    // fmt chunk
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());      // chunk size
    bytes.extend_from_slice(&1u16.to_le_bytes());       // audio format (PCM)
    bytes.extend_from_slice(&num_channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk header
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_size.to_le_bytes());

    // Zero-copy cast: &[i16] -> &[u8]
    let sample_bytes: &[u8] = bytemuck::cast_slice(samples);
    bytes.extend_from_slice(sample_bytes);

    // Single write syscall
    fs::write(path.as_ref(), &bytes)?;

    Ok(())
}
