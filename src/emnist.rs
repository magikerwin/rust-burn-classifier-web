use std::fs::{self, File};
use std::io::{Read, BufReader};
use std::path::Path;
use flate2::read::GzDecoder;
use burn::data::dataset::{vision::MnistItem, Dataset};

pub const EMNIST_CLASSES: [&str; 26] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
];

pub struct EmnistDataset {
    items: Vec<MnistItem>,
}

impl EmnistDataset {
    pub fn new(train: bool) -> Self {
        let cache_dir = Path::new("./target/emnist_dataset");
        fs::create_dir_all(cache_dir).ok();

        let prefix = if train { "train" } else { "test" };
        let img_filename = format!("emnist-letters-{}-images-idx3-ubyte.gz", prefix);
        let lbl_filename = format!("emnist-letters-{}-labels-idx1-ubyte.gz", prefix);

        let img_path = cache_dir.join(&img_filename);
        let lbl_path = cache_dir.join(&lbl_filename);

        // Download files if they do not exist
        if !img_path.exists() {
            println!("Downloading EMNIST Letters {} images...", prefix);
            download_file(&img_filename, &img_path)
                .unwrap_or_else(|e| panic!("Failed to download {}: {}", img_filename, e));
        }

        if !lbl_path.exists() {
            println!("Downloading EMNIST Letters {} labels...", prefix);
            download_file(&lbl_filename, &lbl_path)
                .unwrap_or_else(|e| panic!("Failed to download {}: {}", lbl_filename, e));
        }

        // Parse images and labels
        println!("Loading and parsing EMNIST Letters {} data...", prefix);
        let images = parse_images_gzip(&img_path);
        let labels = parse_labels_gzip(&lbl_path);

        assert_eq!(images.len(), labels.len(), "Images and labels count mismatch!");

        let items = images
            .into_iter()
            .zip(labels.into_iter())
            .map(|(img, lbl)| MnistItem {
                image: img,
                label: lbl,
            })
            .collect();

        Self { items }
    }
}

impl Dataset<MnistItem> for EmnistDataset {
    fn get(&self, index: usize) -> Option<MnistItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

fn download_file(filename: &str, dest_path: &Path) -> Result<(), String> {
    let url = format!(
        "https://huggingface.co/datasets/Heliosoph/EMNIST/resolve/main/{}",
        filename
    );

    let client = reqwest::blocking::Client::new();
    let mut response = client
        .get(&url)
        .send()
        .map_err(|e| format!("HTTP request failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Server returned error status: {}", response.status()));
    }

    let mut dest_file = File::create(dest_path)
        .map_err(|e| format!("Failed to create destination file: {}", e))?;

    response
        .copy_to(&mut dest_file)
        .map_err(|e| format!("Failed to write data to file: {}", e))?;

    Ok(())
}

fn parse_images_gzip(path: &Path) -> Vec<[[f32; 28]; 28]> {
    let file = File::open(path).expect("Failed to open images file");
    let decoder = GzDecoder::new(file);
    let mut reader = BufReader::new(decoder);

    // Read magic number
    let mut magic_buf = [0u8; 4];
    reader.read_exact(&mut magic_buf).expect("Failed to read magic number");
    let magic = u32::from_be_bytes(magic_buf);
    assert_eq!(magic, 2051, "Invalid magic number for images!");

    // Read number of images
    let mut count_buf = [0u8; 4];
    reader.read_exact(&mut count_buf).expect("Failed to read image count");
    let count = u32::from_be_bytes(count_buf) as usize;

    // Read height and width
    let mut rows_buf = [0u8; 4];
    let mut cols_buf = [0u8; 4];
    reader.read_exact(&mut rows_buf).expect("Failed to read rows");
    reader.read_exact(&mut cols_buf).expect("Failed to read columns");
    let rows = u32::from_be_bytes(rows_buf) as usize;
    let cols = u32::from_be_bytes(cols_buf) as usize;
    assert_eq!(rows, 28);
    assert_eq!(cols, 28);

    let mut images = Vec::with_capacity(count);
    let mut img_buf = vec![0u8; 28 * 28];

    for _ in 0..count {
        reader.read_exact(&mut img_buf).expect("Failed to read image pixel data");
        let mut img = [[0.0f32; 28]; 28];
        for y in 0..28 {
            for x in 0..28 {
                // EMNIST images are stored transposed (rotated & flipped).
                // Swapping rows and columns transposes them back to the correct standard orientation.
                img[y][x] = img_buf[x * 28 + y] as f32 / 255.0;
            }
        }
        images.push(img);
    }

    images
}

fn parse_labels_gzip(path: &Path) -> Vec<u8> {
    let file = File::open(path).expect("Failed to open labels file");
    let decoder = GzDecoder::new(file);
    let mut reader = BufReader::new(decoder);

    // Read magic number
    let mut magic_buf = [0u8; 4];
    reader.read_exact(&mut magic_buf).expect("Failed to read magic number");
    let magic = u32::from_be_bytes(magic_buf);
    assert_eq!(magic, 2049, "Invalid magic number for labels!");

    // Read count
    let mut count_buf = [0u8; 4];
    reader.read_exact(&mut count_buf).expect("Failed to read labels count");
    let count = u32::from_be_bytes(count_buf) as usize;

    let mut labels = Vec::with_capacity(count);
    let mut buf = vec![0u8; count];
    reader.read_exact(&mut buf).expect("Failed to read label bytes");

    for raw_label in buf {
        // EMNIST letters labels are 1-indexed (1 to 26 representing A-Z).
        // Normalize them to 0-indexed (0 to 25) to match our classification network targets.
        assert!(raw_label >= 1 && raw_label <= 26, "Label value out of bounds: {}", raw_label);
        labels.push(raw_label - 1);
    }

    labels
}
