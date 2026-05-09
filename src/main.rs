mod binary_load;
mod binary_save;
mod light_network;
mod network;

use light_network::LightLayer;
use network::Layer;
use rand::{self, Rng};
use std::env;
use std::fs::{self, DirEntry, File};
use std::io::{self, Write};
use std::path::Path;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        return run_from_cli(&args);
    }
    loop {
        let mut input_menu = String::new();
        println!(
            "実行したい操作は?
1 新しいモデルを作る
2 モデルを読み込んで自動試験
3 モデルを読み込んで手動試験
4 終了"
        );
        let _ = io::stdin().read_line(&mut input_menu);
        match input_menu.trim().parse::<u8>() {
            Ok(1) => {
                println!(
                    "入力層側から順に、各中間層に何個のニューロンを持たせるか空白区切りで入力して下さい。"
                );
                let mut _layer_sizes: Vec<usize> = Vec::new();
                {
                    loop {
                        let mut input_num_layer = String::new();
                        let _ = io::stdin().read_line(&mut input_num_layer);
                        _layer_sizes = input_num_layer
                            .split_whitespace()
                            .map(|s| s.parse::<usize>().unwrap_or(0))
                            .collect();
                        if _layer_sizes.contains(&0) || _layer_sizes.contains(&1) {
                            println!("予期されていない数値が入力されたので、入力し直して下さい。");
                        } else {
                            break;
                        }
                    }
                }
                //layer_sizesは入力層含む
                //layersは入力層含まない
                _layer_sizes.insert(0, 784);
                _layer_sizes.push(10);
                let num_layers = _layer_sizes.len() - 1;
                let mut layers: Vec<Layer> = Vec::with_capacity(num_layers);
                for i in 0..num_layers {
                    let size_shallower_layer = *_layer_sizes.get(i).unwrap();
                    let size_current_layer = *_layer_sizes.get(i + 1).unwrap();
                    layers.push(Layer::new(
                        size_current_layer,
                        size_shallower_layer,
                        i == num_layers - 1,
                    ));
                }
                let mut image_file = File::open("datas/digits_image.bin")?;
                let num_images = binary_load::get_num_of_images(&mut image_file)?;
                let mut label_file = File::open("datas/digits_label.bin")?;
                let _ = binary_load::get_num_of_labels(&mut label_file)?;

                let mut all_images: Vec<Box<Vec<f64>>> = Vec::with_capacity(num_images);
                for _ in 0..num_images {
                    all_images.push(binary_load::get_next_image(&mut image_file)?);
                }
                let mut all_labels: Vec<u8> = Vec::with_capacity(num_images);
                for _ in 0..num_images {
                    all_labels.push(binary_load::get_next_label(&mut label_file)?);
                }

                let (size_batch, num_epoch, mut learning_rate) = loop {
                    println!("バッチサイズとエポック数、学習率を空白区切りで入力して下さい。");
                    let mut learning_parameters = String::new();
                    let _ = io::stdin().read_line(&mut learning_parameters);
                    let mut learning_parameters = learning_parameters.split_whitespace();
                    if let Ok(size_batch) = learning_parameters.next().unwrap().parse::<usize>() {
                        if let Ok(num_epoch) = learning_parameters.next().unwrap().parse::<usize>()
                        {
                            if let Ok(learning_rate) =
                                learning_parameters.next().unwrap().parse::<f64>()
                            {
                                break (size_batch, num_epoch, learning_rate);
                            } else {
                                print!("学習率");
                            }
                        } else {
                            print!("エポック数");
                        }
                    } else {
                        print!("バッチサイズ");
                    }
                    println!("の値が不正です。再度入力して下さい。");
                };
                let num_iteration: usize = num_images / size_batch;
                //学習の進み具合に伴って学習率を小さくしていく為の係数
                let learning_rate_coefficient =
                    100f64.powf(1.0 / (num_iteration * num_epoch) as f64);
                let mut rng = rand::thread_rng();
                let mut cost = 0.0;
                let mut label_debug: usize = 0;
                for epoch in 0..num_epoch {
                    for iteration in 0..num_iteration {
                        for _batch in 0..size_batch {
                            let data_index = rng.gen_range(0..num_images);
                            let image = all_images.get(data_index).unwrap();
                            let label = all_labels.get(data_index).unwrap();
                            label_debug = *label as usize;
                            cost = network::backpropagation(
                                &mut layers,
                                image,
                                learning_rate,
                                *label as usize,
                            );
                        }
                        print!("\r\x1b[K");
                        io::stdout().flush()?;
                        print!(
                            "cost: {:.4}\titeration: {}/{}\tepoch: {}/{}\tlearning rate: {:.8}\tanswer: {}\toutputs: ",
                            cost,
                            iteration,num_iteration,
                            epoch,num_epoch,
                            learning_rate,
                            label_debug
                        );
                        for value in layers.last().unwrap().get_neurons_activations() {
                            print!(" {:.3} ", value);
                        }
                        io::stdout().flush().unwrap();
                        network::apply_neurons_fixes(&mut layers, size_batch);
                        learning_rate /= learning_rate_coefficient;
                    }
                }
                println!();
                if let Err(x) = binary_save::save_model(&layers, &_layer_sizes[1..]) {
                    println!("モデルの保存中にエラーが発生しました:\n{}", x);
                    return Err(x);
                } else {
                    println!("正常にモデルを保存できました。");
                }
            }
            Ok(2) => {
                let layer_sizes_len: usize;
                let layer_sizes: Vec<usize>;
                println!("読み込むモデルの名前を入力して下さい。");
                let mut model_name = String::new();
                'input_model_name: loop {
                    let _ = io::stdin().read_line(&mut model_name);
                    if let Ok(mut header_file) =
                        File::open(format!("save_datas/{}/header.bin", model_name.trim()))
                    {
                        if let Ok((number, sizes)) = binary_load::load_header(&mut header_file) {
                            layer_sizes_len = number;
                            layer_sizes = sizes;
                            break 'input_model_name;
                        } else {
                            panic!("モデルの読み込み中にエラーが発生しました。");
                        }
                    } else {
                        println!(
                            "その名前のモデルは存在しないかも知れません。もう一度入力して下さい。"
                        );
                        model_name.clear();
                    }
                }
                let model_name = model_name.trim().to_string();
                let mut layers: Vec<LightLayer> = Vec::with_capacity(layer_sizes_len);
                for i in 0..layer_sizes_len {
                    let path = Path::new("save_datas")
                        .join(&model_name)
                        .join(format!("layer{}.bin", i));
                    let mut file = File::open(&path)?;
                    layers.push(LightLayer::new(
                        &mut file,
                        *layer_sizes.get(i).unwrap(),
                        if i == 0 {
                            784
                        } else {
                            *layer_sizes.get(i - 1).unwrap()
                        },
                    ));
                }
                let mut test_image_file = File::open("datas/digits_test_image.bin")?;
                let mut test_label_file = File::open("datas/digits_test_label.bin")?;
                let num_test_images = binary_load::get_num_of_images(&mut test_image_file)?;
                let num_test_labels = binary_load::get_num_of_labels(&mut test_label_file)?;

                let mut num_correct: u32 = 0;
                let mut cost = 0f64;
                for index_of_images in 0..num_test_images {
                    print!("image: {}/{}", index_of_images, num_test_images);
                    io::stdout().flush().unwrap();
                    let test_image = binary_load::get_next_image(&mut test_image_file)?;
                    let test_label = binary_load::get_next_label(&mut test_label_file)?;
                    light_network::guess_answer(&mut layers, &test_image);
                    let last_layer_activations = layers.last().unwrap().get_neurons_activations();
                    let answer_of_network =
                        last_layer_activations
                            .iter()
                            .enumerate()
                            .fold(0, |max_i, (i, &x)| {
                                if x > last_layer_activations[max_i] {
                                    i
                                } else {
                                    max_i
                                }
                            });
                    cost -= last_layer_activations.get(answer_of_network).unwrap().ln();
                    if test_label == answer_of_network as u8 {
                        num_correct += 1;
                    }
                    print!("\r\x1b[K");
                    io::stdout().flush().unwrap();
                }
                cost /= num_test_labels as f64;
                println!(
                    "cost: {:.5}\tcorrect answer rate: {:.1}%",
                    cost,
                    100f64 * (num_correct as f64 / num_test_labels as f64)
                );
            }
            Ok(3) => {
                let layer_sizes_len: usize;
                let layer_sizes: Vec<usize>;
                println!("読み込むモデルの名前を入力して下さい。");
                let mut model_name = String::new();
                'input_model_name: loop {
                    let _ = io::stdin().read_line(&mut model_name);
                    if let Ok(mut header_file) =
                        File::open(format!("save_datas/{}/header.bin", model_name.trim()))
                    {
                        if let Ok((number, sizes)) = binary_load::load_header(&mut header_file) {
                            layer_sizes_len = number;
                            layer_sizes = sizes;
                            break 'input_model_name;
                        } else {
                            panic!("モデルの読み込み中にエラーが発生しました。");
                        }
                    } else {
                        println!(
                            "その名前のモデルは存在しないかも知れません。もう一度入力して下さい。"
                        );
                        model_name.clear();
                    }
                }
                let model_name = model_name.trim().to_string();
                let mut layers: Vec<LightLayer> = Vec::with_capacity(layer_sizes_len);
                for i in 0..layer_sizes_len {
                    let path = Path::new("save_datas")
                        .join(&model_name)
                        .join(format!("layer{}.bin", i));
                    let mut file = File::open(&path)?;
                    layers.push(LightLayer::new(
                        &mut file,
                        *layer_sizes.get(i).unwrap(),
                        if i == 0 {
                            784
                        } else {
                            *layer_sizes.get(i - 1).unwrap()
                        },
                    ));
                }

                let folder_path = Path::new("test_image");
                let entries: Vec<DirEntry> = fs::read_dir(folder_path)?
                    .filter_map(Result::ok)
                    .filter(|e| e.path().is_file())
                    .collect();
                let mut image_file: File;
                match entries.len() {
                    0 => {
                        println!("読み込めるファイルが見つかりませんでした。");
                        continue;
                    }
                    1 => {
                        if let Ok(file) = File::open(entries[0].path()) {
                            image_file = file;
                        } else {
                            println!("ファイル読み込み時にエラーが発生しました。");
                            continue;
                        }
                    }
                    x => {
                        println!("{}個のファイルが見つかりました。\n0~{}の数字を入力し、ファイルを選んで下さい。",x,x-1);
                        for (i, entry) in entries.iter().enumerate() {
                            println!("{}: {:?}", i, entry.path().file_name().unwrap());
                        }
                        let choice: usize;
                        loop {
                            let mut input = String::new();
                            let _ = io::stdin().read_line(&mut input);
                            if let Ok(choice_i) = input.trim().parse::<usize>() {
                                if (0..x).contains(&choice_i) {
                                    choice = choice_i;
                                    break;
                                } else {
                                    print!("0~{}の範囲内で", x - 1);
                                }
                            } else {
                                println!("変な文字が入力されました。");
                            }
                            println!("入力し直して下さい。");
                        }
                        if let Ok(file) = File::open(entries[choice].path()) {
                            image_file = file;
                        } else {
                            println!("ファイル読み込み時にエラーが発生しました。");
                            continue;
                        }
                    }
                }
                let image = binary_load::get_next_image(&mut image_file)?;
                light_network::guess_answer(&mut layers, &image);
                let last_layer_activations = layers.last().unwrap().get_neurons_activations();
                let mut indexed_last_layer_activations: Vec<(usize, f64)> =
                    last_layer_activations.into_iter().enumerate().collect();
                indexed_last_layer_activations.sort_by(|i, o| i.1.partial_cmp(&o.1).unwrap());
                let answer_of_network = indexed_last_layer_activations.last().unwrap().0;
                for (i, value) in image.iter().enumerate() {
                    print!("{}", if *value > 0.5 { "# " } else { "  " });
                    if i % 28 == 0 {
                        println!();
                    }
                }
                println!();
                for (index, output) in indexed_last_layer_activations.into_iter().rev() {
                    println!("{} である確率: {: >7.3} %", index, output * 100f64);
                }
                println!("\n予測: これは {} です!", answer_of_network);
            }
            Ok(4) => {
                println!("終了します。");
                break;
            }
            Ok(_) => println!("1~4の数字を入力してください。"),
            Err(_) => println!("謎の文字を入力しないでください。"),
        }
        println!();
    }
    Ok(())
}

fn run_from_cli(args: &[String]) -> io::Result<()> {
    match args[1].as_str() {
        "train" | "train-test" => {
            let hidden_layers = parse_usize_csv(get_arg_value(args, "--layers")?)?;
            let batch_size = get_arg_value(args, "--batch-size")?.parse::<usize>().map_err(invalid_input)?;
            let epochs = get_arg_value(args, "--epochs")?.parse::<usize>().map_err(invalid_input)?;
            let learning_rate = get_arg_value(args, "--learning-rate")?.parse::<f64>().map_err(invalid_input)?;
            let model_name = get_arg_value(args, "--model-name")?;
            let epoch_log = find_optional_arg_value(args, "--epoch-log");
            train_model(hidden_layers, batch_size, epochs, learning_rate, model_name, epoch_log)?;
            if args[1] == "train-test" {
                run_auto_test(model_name)?;
            }
            Ok(())
        }
        "test" => run_auto_test(get_arg_value(args, "--model-name")?),
        _ => Err(io::Error::new(io::ErrorKind::InvalidInput, "不明なサブコマンドです。")),
    }
}

fn get_arg_value<'a>(args: &'a [String], key: &str) -> io::Result<&'a str> {
    let index = args.iter().position(|arg| arg == key).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("{key} が指定されていません。")))?;
    args.get(index + 1).map(|s| s.as_str()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("{key} の値が不足しています。")))
}
fn find_optional_arg_value<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter().position(|arg| arg == key).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn parse_usize_csv(csv: &str) -> io::Result<Vec<usize>> {
    let mut values = Vec::new();
    for token in csv.split(',') {
        let n = token.trim().parse::<usize>().map_err(invalid_input)?;
        if n <= 1 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "層のニューロン数は2以上で指定してください。"));
        }
        values.push(n);
    }
    Ok(values)
}

fn invalid_input<E: std::fmt::Display>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, e.to_string())
}

fn train_model(hidden_layers: Vec<usize>, size_batch: usize, num_epoch: usize, mut learning_rate: f64, model_name: &str, epoch_log: Option<&str>) -> io::Result<()> {
    let mut layer_sizes = hidden_layers;
    layer_sizes.insert(0, 784);
    layer_sizes.push(10);
    let num_layers = layer_sizes.len() - 1;
    let mut layers: Vec<Layer> = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        layers.push(Layer::new(layer_sizes[i + 1], layer_sizes[i], i == num_layers - 1));
    }
    let mut image_file = File::open("datas/digits_image.bin")?;
    let num_images = binary_load::get_num_of_images(&mut image_file)?;
    let mut label_file = File::open("datas/digits_label.bin")?;
    let _ = binary_load::get_num_of_labels(&mut label_file)?;
    let mut all_images: Vec<Box<Vec<f64>>> = Vec::with_capacity(num_images);
    for _ in 0..num_images { all_images.push(binary_load::get_next_image(&mut image_file)?); }
    let mut all_labels: Vec<u8> = Vec::with_capacity(num_images);
    for _ in 0..num_images { all_labels.push(binary_load::get_next_label(&mut label_file)?); }
    let num_iteration = num_images / size_batch;
    let learning_rate_coefficient = 100f64.powf(1.0 / (num_iteration * num_epoch) as f64);
    let mut epoch_writer = if let Some(path) = epoch_log {
        let mut file = File::create(path)?;
        writeln!(file, "model_name,epoch,cost,accuracy")?;
        Some(file)
    } else {
        None
    };
    let mut rng = rand::thread_rng();
    for epoch in 0..num_epoch {
        for _iteration in 0..num_iteration {
            for _batch in 0..size_batch {
                let data_index = rng.gen_range(0..num_images);
                let image = all_images.get(data_index).unwrap();
                let label = all_labels.get(data_index).unwrap();
                let _ = network::backpropagation(&mut layers, image, learning_rate, *label as usize);
            }
            network::apply_neurons_fixes(&mut layers, size_batch);
            learning_rate /= learning_rate_coefficient;
        }
        if let Some(file) = epoch_writer.as_mut() {
            let (cost, acc) = evaluate_with_test_data(&mut layers)?;
            writeln!(file, "{},{},{:.6},{:.4}", model_name, epoch + 1, cost, acc)?;
        }
    }
    binary_save::save_model_with_name(&layers, &layer_sizes[1..], model_name)
}
fn evaluate_with_test_data(layers: &mut Vec<Layer>) -> io::Result<(f64, f64)> {
    let mut test_image_file = File::open("datas/digits_test_image.bin")?;
    let mut test_label_file = File::open("datas/digits_test_label.bin")?;
    let num_test_images = binary_load::get_num_of_images(&mut test_image_file)?;
    let num_test_labels = binary_load::get_num_of_labels(&mut test_label_file)?;
    let mut num_correct: u32 = 0;
    let mut cost = 0f64;
    for _ in 0..num_test_images {
        let test_image = binary_load::get_next_image(&mut test_image_file)?;
        let test_label = binary_load::get_next_label(&mut test_label_file)?;
        network::guess_answer(layers, &test_image);
        let activations = layers.last().unwrap().get_neurons_activations();
        let answer = activations.iter().enumerate().fold(0, |max_i, (i, &x)| if x > activations[max_i] { i } else { max_i });
        cost -= activations[answer].ln();
        if test_label == answer as u8 { num_correct += 1; }
    }
    cost /= num_test_labels as f64;
    let accuracy = 100f64 * (num_correct as f64 / num_test_labels as f64);
    Ok((cost, accuracy))
}
fn run_auto_test(model_name: &str) -> io::Result<()> {
    let mut header_file = File::open(format!("save_datas/{}/header.bin", model_name))?;
    let (layer_sizes_len, layer_sizes) = binary_load::load_header(&mut header_file)?;
    let mut layers: Vec<LightLayer> = Vec::with_capacity(layer_sizes_len);
    for i in 0..layer_sizes_len {
        let path = Path::new("save_datas").join(model_name).join(format!("layer{}.bin", i));
        let mut file = File::open(&path)?;
        layers.push(LightLayer::new(&mut file, layer_sizes[i], if i == 0 { 784 } else { layer_sizes[i - 1] }));
    }
    let mut test_image_file = File::open("datas/digits_test_image.bin")?;
    let mut test_label_file = File::open("datas/digits_test_label.bin")?;
    let num_test_images = binary_load::get_num_of_images(&mut test_image_file)?;
    let num_test_labels = binary_load::get_num_of_labels(&mut test_label_file)?;
    let mut num_correct: u32 = 0;
    let mut cost = 0f64;
    for _ in 0..num_test_images {
        let test_image = binary_load::get_next_image(&mut test_image_file)?;
        let test_label = binary_load::get_next_label(&mut test_label_file)?;
        light_network::guess_answer(&mut layers, &test_image);
        let activations = layers.last().unwrap().get_neurons_activations();
        let answer = activations.iter().enumerate().fold(0, |max_i, (i, &x)| if x > activations[max_i] { i } else { max_i });
        cost -= activations[answer].ln();
        if test_label == answer as u8 { num_correct += 1; }
    }
    cost /= num_test_labels as f64;
    let accuracy = 100f64 * (num_correct as f64 / num_test_labels as f64);
    println!("model={model_name} cost={cost:.5} accuracy={accuracy:.2}%");
    Ok(())
}
