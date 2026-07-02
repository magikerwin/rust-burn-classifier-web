use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use burn::data::dataset::{vision::MnistItem, Dataset};

pub const QUICKDRAW_CLASSES: [&str; 25] = [
    "triangle", "circle", "square", "sun", "moon", "star", "tree", "flower",
    "cat", "dog", "fish", "butterfly", "cup", "key", "umbrella", "hat",
    "clock", "envelope", "toothbrush", "house", "car", "smiley face", "pants", "t-shirt", "apple"
];

pub const TRAIN_SAMPLES_PER_CLASS: usize = 2000;
pub const VAL_SAMPLES_PER_CLASS: usize = 500;
pub const TOTAL_SAMPLES_PER_CLASS: usize = TRAIN_SAMPLES_PER_CLASS + VAL_SAMPLES_PER_CLASS;

/// Dataset struct for Quick, Draw! items.
/// Reuses MnistItem to allow full compatibility with MnistBatcher.
pub struct QuickDrawDataset {
    items: Vec<MnistItem>,
}

impl QuickDrawDataset {
    /// Loads a subset of categories for either training or testing.
    /// - `train`: If true, loads train samples, otherwise loads test samples.
    /// - `samples_per_class`: Number of samples to load per class for the requested set.
    pub fn new(train: bool, samples_per_class: usize) -> Self {
        let cache_dir = Path::new("./target/quickdraw_dataset");
        fs::create_dir_all(cache_dir).ok();

        let mut items = Vec::new();
        // For each class, we download the total required samples
        let total_needed = TOTAL_SAMPLES_PER_CLASS;

        for (label_idx, class_name) in QUICKDRAW_CLASSES.iter().enumerate() {
            let file_path = cache_dir.join(format!("{}.npy", class_name));
            
            // 1. Download if not cached
            if !file_path.exists() {
                println!("Downloading Quick, Draw! category: '{}'...", class_name);
                download_class_subset(class_name, total_needed, &file_path)
                    .unwrap_or_else(|e| panic!("Failed to download class '{}': {}", class_name, e));
            }

            // 2. Load and parse from file
            let mut file = File::open(&file_path)
                .unwrap_or_else(|e| panic!("Failed to open cached file for '{}': {}", class_name, e));
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)
                .unwrap_or_else(|e| panic!("Failed to read cached file for '{}': {}", class_name, e));

            let images = parse_npy(&buffer)
                .unwrap_or_else(|e| panic!("Failed to parse npy for '{}': {}", class_name, e));

            // 3. Slice the training/validation partition
            let (start_idx, end_idx) = if train {
                (0, samples_per_class.min(TRAIN_SAMPLES_PER_CLASS))
            } else {
                (
                    TRAIN_SAMPLES_PER_CLASS,
                    (TRAIN_SAMPLES_PER_CLASS + samples_per_class).min(TOTAL_SAMPLES_PER_CLASS),
                )
            };

            for img in images.into_iter().skip(start_idx).take(end_idx - start_idx) {
                items.push(MnistItem {
                    image: img,
                    label: label_idx as u8,
                });
            }
        }

        Self { items }
    }
}

impl Dataset<MnistItem> for QuickDrawDataset {
    fn get(&self, index: usize) -> Option<MnistItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

/// Helper to download the first N samples from Google Cloud Storage using HTTP Range.
fn download_class_subset(class_name: &str, num_samples: usize, dest_path: &Path) -> Result<(), String> {
    // 2500 samples * 784 bytes ≈ 1.96 MB. Requesting 2.1 MB to safely cover any header size.
    let bytes_to_request = (num_samples * 784) + 128 * 1024; 
    let encoded_class = class_name.replace(" ", "%20");
    let url = format!(
        "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy",
        encoded_class
    );

    let client = reqwest::blocking::Client::new();
    let response = client
        .get(&url)
        .header("Range", format!("bytes=0-{}", bytes_to_request - 1))
        .send()
        .map_err(|e| format!("Request failed: {}", e))?;

    if !response.status().is_success() && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(format!("Server returned error status: {}", response.status()));
    }

    let bytes = response.bytes().map_err(|e| format!("Failed to read bytes: {}", e))?;
    let mut file = File::create(dest_path).map_err(|e| format!("Failed to create file: {}", e))?;
    file.write_all(&bytes).map_err(|e| format!("Failed to write to file: {}", e))?;

    Ok(())
}

/// Simple parser to extract 28x28 grayscale normalized arrays from a `.npy` file.
fn parse_npy(bytes: &[u8]) -> Result<Vec<[[f32; 28]; 28]>, String> {
    if bytes.len() < 10 {
        return Err("File too short".into());
    }
    if &bytes[0..6] != b"\x93NUMPY" {
        return Err("Not a valid numpy file format".into());
    }
    let major = bytes[6];
    let header_len = if major == 1 {
        u16::from_le_bytes([bytes[8], bytes[9]]) as usize
    } else if major == 2 || major == 3 {
        if bytes.len() < 12 {
            return Err("File too short for header length".into());
        }
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize
    } else {
        return Err(format!("Unsupported numpy major version: {}", major));
    };

    let offset = if major == 1 { 10 + header_len } else { 12 + header_len };
    if bytes.len() < offset {
        return Err("File truncated before data start".into());
    }

    let data_bytes = &bytes[offset..];
    let num_images = data_bytes.len() / 784;
    let mut images = Vec::with_capacity(num_images);

    for i in 0..num_images {
        let start = i * 784;
        let mut grid = [[0.0f32; 28]; 28];
        for r in 0..28 {
            for c in 0..28 {
                let pixel_val = data_bytes[start + r * 28 + c];
                grid[r][c] = pixel_val as f32 / 255.0; // Normalize pixel values to 0.0..1.0
            }
        }
        images.push(grid);
    }

    Ok(images)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_mock_npy() {
        // We build a mock .npy file in memory to test our parser without downloading real files.
        let mut npy_data = Vec::new();

        // Step 1: Write the official NumPy signature (6 bytes) and version 1.0 (2 bytes)
        npy_data.extend_from_slice(b"\x93NUMPY\x01\x00");
        
        // Step 2: Define metadata as a Python dictionary string:
        // - 'descr': '<u1' means 1-byte unsigned integer (uint8/u8 pixels)
        // - 'shape': (1, 784) means 1 image of 784 pixels (28x28)
        let header_str = "{'descr': '<u1', 'fortran_order': False, 'shape': (1, 784)}";
        let mut header_bytes = header_str.as_bytes().to_vec();

        // Step 3: Pad the header string with spaces so the total header size
        // (10 bytes prefix + header length + newline character) is divisible by 64.
        while (10 + header_bytes.len() + 1) % 64 != 0 {
            header_bytes.push(b' ');
        }
        header_bytes.push(b'\n'); // Terminate the header with a newline
        
        // Step 4: Write the 2-byte header length (little-endian) followed by the header string
        let header_len = header_bytes.len();
        npy_data.extend_from_slice(&(header_len as u16).to_le_bytes());
        npy_data.extend_from_slice(&header_bytes);
        
        // Step 5: Append raw pixel data (784 bytes of value 255 representing a fully white image)
        npy_data.extend(std::iter::repeat(255).take(784));

        // Step 6: Verify our parser correctly extracts 1 image and normalizes pixel values to 1.0 (255 / 255)
        let images = parse_npy(&npy_data).expect("Parsing mock npy failed");
        assert_eq!(images.len(), 1);
        assert_eq!(images[0][0][0], 1.0f32);
        assert_eq!(images[0][27][27], 1.0f32);
    }

    #[test]
    fn test_quickdraw_dataset_loading() {
        use std::fs;
        let cache_dir = Path::new("./target/quickdraw_dataset");
        fs::create_dir_all(cache_dir).ok();

        // Construct a helper to generate minimal valid mock .npy bytes representing N samples
        let make_mock_npy = |num_samples: usize| -> Vec<u8> {
            let mut npy_data = Vec::new();
            npy_data.extend_from_slice(b"\x93NUMPY\x01\x00");
            let header_str = format!("{{'descr': '<u1', 'fortran_order': False, 'shape': ({}, 784)}}", num_samples);
            let mut header_bytes = header_str.as_bytes().to_vec();
            while (10 + header_bytes.len() + 1) % 64 != 0 {
                header_bytes.push(b' ');
            }
            header_bytes.push(b'\n');
            let header_len = header_bytes.len();
            npy_data.extend_from_slice(&(header_len as u16).to_le_bytes());
            npy_data.extend_from_slice(&header_bytes);
            npy_data.extend(std::iter::repeat(128).take(num_samples * 784));
            npy_data
        };

        // Write mock files for all 25 classes if they don't exist, so the test doesn't trigger network downloads
        let mut created_paths = Vec::new();
        for class_name in QUICKDRAW_CLASSES.iter() {
            let path = cache_dir.join(format!("{}.npy", class_name));
            if !path.exists() {
                let data = make_mock_npy(TOTAL_SAMPLES_PER_CLASS);
                fs::write(&path, data).unwrap();
                created_paths.push(path);
            }
        }

        // Test training set construction (request 5 samples per class)
        let train_dataset = QuickDrawDataset::new(true, 5);
        assert_eq!(train_dataset.len(), QUICKDRAW_CLASSES.len() * 5);
        let sample = train_dataset.get(0).unwrap();
        assert_eq!(sample.label, 0);

        // Test validation set construction (request 2 samples per class)
        let valid_dataset = QuickDrawDataset::new(false, 2);
        assert_eq!(valid_dataset.len(), QUICKDRAW_CLASSES.len() * 2);

        // Clean up only the mock files created during the test
        for path in created_paths {
            fs::remove_file(path).ok();
        }
    }
}
