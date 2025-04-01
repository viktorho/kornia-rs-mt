use std::path::Path;

use kornia_image::{Image, ImageSize};

use crate::error::IoError;

#[cfg(feature = "turbojpeg")]
use super::jpegturbo::{JpegTurboDecoder, JpegTurboEncoder};

#[cfg(feature = "turbojpeg")]
/// Reads a JPEG image in `RGB8` format from the given file path.
///
/// The method reads the JPEG image data directly from a file leveraging the libjpeg-turbo library.
///
/// # Arguments
///
/// * `image_path` - The path to the JPEG image.
///
/// # Returns
///
/// An in image containing the JPEG image data.
///
/// # Example
///
/// ```
/// use kornia_image::Image;
/// use kornia_io::functional as F;
///
/// let image: Image<u8, 3> = F::read_image_jpegturbo_rgb8("../../tests/data/dog.jpeg").unwrap();
///
/// assert_eq!(image.cols(), 258);
/// assert_eq!(image.rows(), 195);
/// assert_eq!(image.num_channels(), 3);
/// ```
pub fn read_image_jpegturbo_rgb8(file_path: impl AsRef<Path>) -> Result<Image<u8, 3>, IoError> {
    let file_path = file_path.as_ref().to_owned();
    // verify the file exists and is a JPEG
    if !file_path.exists() {
        return Err(IoError::FileDoesNotExist(file_path.to_path_buf()));
    }

    if file_path.extension().map_or(true, |ext| {
        !ext.eq_ignore_ascii_case("jpg") && !ext.eq_ignore_ascii_case("jpeg")
    }) {
        return Err(IoError::InvalidFileExtension(file_path.to_path_buf()));
    }

    // open the file and map it to memory
    let jpeg_data = std::fs::read(file_path)?;

    // decode the data directly from memory
    let image: Image<u8, 3> = {
        let mut decoder = JpegTurboDecoder::new()?;
        decoder.decode_rgb8(&jpeg_data)?
    };

    Ok(image)
}

#[cfg(feature = "turbojpeg")]
/// Writes the given JPEG data to the given file path.
///
/// # Arguments
///
/// * `file_path` - The path to the JPEG image.
/// * `image` - The tensor containing the JPEG image data.
pub fn write_image_jpegturbo_rgb8(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 3>,
) -> Result<(), IoError> {
    let file_path = file_path.as_ref().to_owned();

    // compress the image
    let jpeg_data = JpegTurboEncoder::new()?.encode_rgb8(image)?;

    // write the data directly to a file
    std::fs::write(file_path, jpeg_data)?;

    Ok(())
}

/// Reads a RGB8 image from the given file path.
///
/// The method tries to read from any image format supported by the image crate.
///
/// # Arguments
///
/// * `file_path` - The path to the image.
///
/// # Returns
///
/// A tensor image containing the image data in RGB8 format with shape (H, W, 3).
///
/// # Example
///
/// ```
/// use kornia_image::Image;
/// use kornia_io::functional as F;
///
/// let image: Image<u8, 3> = F::read_image_any_rgb8("../../tests/data/dog.jpeg").unwrap();
///
/// assert_eq!(image.cols(), 258);
/// assert_eq!(image.rows(), 195);
/// assert_eq!(image.num_channels(), 3);
/// ```
// Định nghĩa trait ChannelCount để cung cấp hằng số số kênh
pub trait ChannelCount {
    const COUNT: usize;
}

// Một kiểu cụ thể cho 3 kênh, ví dụ dành cho ảnh RGB.
pub struct Channels3;

impl ChannelCount for Channels3 {
    const COUNT: usize = 3;
}

pub trait PixelFormat {
    type ImageType;
    const CHANNELS: usize;
    fn image_reader(file_path: impl AsRef<Path>) -> Result<Image<Self::ImageType,  Self::CHANNELS >, IoError>
    where
        [(); Self::CHANNELS]:;
}


macro_rules! impl_pixel_format {
    ($pf:ident, $img_ty:ty, $channels:expr) => {
        impl PixelFormat for $pf {
            type ImageType = $img_ty;
            type Channels = Channels<$channels>;
            
            fn image_reader(
                file_path: impl AsRef<Path>
            ) ->  Result<Image<Self::ImageType, { Self::Channels::COUNT }>, IoError> {
                // Here we call the utility function to read the PNG file.
                read_png_impl(file_path).map(|(buf, size)| {
                    ImageVar::new(size.into(), buf).unwrap()
                })
            }
        }
    };
}

impl_pixel_format!(Mono8, u8, 1);
impl_pixel_format!(Rgb8, u8, 3);
impl_pixel_format!(Rgba8, u8, 4); 


#[cfg(test)]
mod tests {
    use crate::error::IoError;
    use crate::functional::read_image_any_rgb8;

    #[cfg(feature = "turbojpeg")]
    use crate::functional::{read_image_jpegturbo_rgb8, write_image_jpegturbo_rgb8};

    #[test]
    fn read_any() -> Result<(), IoError> {
        let image = read_image_any_rgb8("../../tests/data/dog.jpeg")?;
        assert_eq!(image.cols(), 258);
        assert_eq!(image.rows(), 195);
        Ok(())
    }

    #[test]
    #[cfg(feature = "turbojpeg")]
    fn read_jpeg() -> Result<(), IoError> {
        let image = read_image_jpegturbo_rgb8("../../tests/data/dog.jpeg")?;
        assert_eq!(image.cols(), 258);
        assert_eq!(image.rows(), 195);
        Ok(())
    }

    #[test]
    #[cfg(feature = "turbojpeg")]
    fn read_write_jpeg() -> Result<(), IoError> {
        let tmp_dir = tempfile::tempdir()?;
        std::fs::create_dir_all(tmp_dir.path())?;

        let file_path = tmp_dir.path().join("dog.jpeg");
        let image_data = read_image_jpegturbo_rgb8("../../tests/data/dog.jpeg")?;
        write_image_jpegturbo_rgb8(&file_path, &image_data)?;

        let image_data_back = read_image_jpegturbo_rgb8(&file_path)?;
        assert!(file_path.exists(), "File does not exist: {:?}", file_path);

        assert_eq!(image_data_back.cols(), 258);
        assert_eq!(image_data_back.rows(), 195);
        assert_eq!(image_data_back.num_channels(), 3);

        Ok(())
    }
}
