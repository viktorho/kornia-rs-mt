use image::DynamicImage;
use kornia_image::{Image, ImageSize}; // Assuming these are defined in your crate
use crate::error::IoError;

/// A trait for converting a `DynamicImage` into an `Image<T, 3>`.
pub trait PixelType: Sized {
    /// Convert a `DynamicImage` into an `Image<T, 3>`.
    fn from_dynamic_image(img: DynamicImage) -> Result<Image<Self, 3>, IoError>;
}

// Implementation for 8-bit pixels.
impl PixelType for u8 {
    fn from_dynamic_image(img: DynamicImage) -> Result<Image<u8, 3>, IoError> {
        // If the decoded image is already in 8-bit RGB, use it directly.
        // Otherwise, force conversion to 8-bit.
        let rgb_img = match img {
            DynamicImage::ImageRgb8(rgb) => rgb,
            other => other.into_rgb8(),
        };
        Image::new(
            ImageSize {
                width: rgb_img.width() as usize,
                height: rgb_img.height() as usize,
            },
            rgb_img.to_vec(),
        )
        .map_err(|e| IoError::ImageConversionError(format!("{}", e)))
    }
}

// Implementation for 32-bit floating point pixels.
impl PixelType for f32 {
    fn from_dynamic_image(img: DynamicImage) -> Result<Image<f32, 3>, IoError> {
        // If the image is already in 32-bit float RGB, use it;
        // otherwise, convert to RGB32F (this scales u8 images into [0.0,1.0]).
        let rgb_img = match img {
            DynamicImage::ImageRgb32F(rgb) => rgb,
            other => other.into_rgb32f(),
        };
        Image::new(
            ImageSize {
                width: rgb_img.width() as usize,
                height: rgb_img.height() as usize,
            },
            rgb_img.to_vec(),
        )
        .map_err(|e| IoError::ImageConversionError(format!("{}", e)))
    }
}
