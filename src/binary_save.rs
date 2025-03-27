use rand_distr::num_traits::ToBytes;

use crate::network::Layer;
use std::fs::{create_dir_all, File};
use std::io::{self, Write};
use std::path::Path;

#[inline]
pub fn save_model(layers: &[Layer], layer_sizes: &[usize]) -> io::Result<()> {
    let mut model_name = String::new();
    println!("保存するフォルダ名を決めて下さい。");
    loop {
        let _ = io::stdin().read_line(&mut model_name)?;
        model_name = model_name.trim().to_string();
        let dir_path = Path::new("save_datas").join(&model_name);
        if create_dir_all(&dir_path).is_ok() {
            println!("{}という名前でモデルを保存しています...", model_name);
            break;
        } else {
            println!("そのフォルダ名は無効です。\nもう一度入力して下さい。");
            model_name.clear();
        }
    }

    //ヘッダー書式: レイヤー数,各層の大きさ浅い方から出力層まで
    let mut path = Path::new("save_datas").join(&model_name).join("header.bin");
    let mut header_file = File::create(&path)?;
    header_file.write_all(&layer_sizes.len().to_be_bytes())?;
    for &size in layer_sizes {
        header_file.write_all(&size.to_be_bytes())?;
    }

    // データ書式: 各バイアス,各重み行列
    for (i, layer) in layers.iter().enumerate() {
        path = Path::new("save_datas")
            .join(&model_name)
            .join(format!("layer{}.bin", i));
        let mut file = File::create(&path)?;
        for neuron in &layer.neurons {
            let (weights, bias) = neuron.get_parameters();
            for weight in weights {
                file.write_all(&weight.to_be_bytes())?;
            }
            let _ = file.write_all(&bias.to_be_bytes());
        }
    }

    Ok(())
}
