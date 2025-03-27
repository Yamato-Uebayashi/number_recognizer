use std::{
    fs::File,
    io::{Error, Read},
};

pub fn get_num_of_images(file: &mut File) -> Result<usize, Error> {
    let mut header = [0u8; 16];
    file.read_exact(&mut header)?;
    let _ = u32::from_be_bytes(header[..4].try_into().unwrap());
    let number = u32::from_be_bytes(header[4..8].try_into().unwrap());
    let _ = u64::from_be_bytes(header[8..].try_into().unwrap());
    Ok(number as usize)
}

pub fn get_next_image(file: &mut File) -> Result<Box<Vec<f64>>, Error> {
    let mut pixels = [0u8; 784];
    let mut normalized_pixels = vec![0f64; 784];
    file.read_exact(&mut pixels)?;
    for col in 0..28 {
        for row in 0..28 {
            normalized_pixels[col * 28 + row] = pixels[row * 28 + col] as f64 / 255f64;
        }
    }
    Ok(Box::new(normalized_pixels))
}

pub fn get_num_of_labels(file: &mut File) -> Result<usize, Error> {
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;
    let _ = u32::from_be_bytes(header[..4].try_into().unwrap());
    Ok(u32::from_be_bytes(header[4..].try_into().unwrap()) as usize)
}

pub fn get_next_label(file: &mut File) -> Result<u8, Error> {
    let mut label = [0u8; 1];
    file.read_exact(&mut label)?;
    Ok(label[0])
}

pub fn load_header(file: &mut File) -> Result<(usize, Vec<usize>), Error> {
    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;
    let num_of_layer = usize::from_be_bytes(header[..].try_into().unwrap());
    let mut layer_sizes_buffer: Vec<u8> = vec![0u8; num_of_layer * 8];
    file.read_exact(&mut layer_sizes_buffer)?;
    let mut layer_sizes: Vec<usize> = vec![0usize; num_of_layer];
    for i in 0..num_of_layer {
        *layer_sizes.get_mut(i).unwrap() =
            usize::from_be_bytes(layer_sizes_buffer[i * 8..(i + 1) * 8].try_into().unwrap());
    }
    Ok((num_of_layer, layer_sizes))
}

#[inline]
pub fn load_neuron(file: &mut File, shallower_size: usize) -> Result<(Vec<f64>, f64), Error> {
    let mut datas: Vec<u8> = vec![0u8; 8 * (1 + shallower_size)];
    file.read_exact(&mut datas)?;
    let mut weights: Vec<f64> = vec![0f64; shallower_size];
    for i in 0..shallower_size {
        *weights.get_mut(i).unwrap() =
            f64::from_be_bytes(datas[i * 8..(i + 1) * 8].try_into().unwrap());
    }
    let bias = f64::from_be_bytes(
        datas[shallower_size * 8..(shallower_size + 1) * 8]
            .try_into()
            .unwrap(),
    );
    Ok((weights, bias))
}
