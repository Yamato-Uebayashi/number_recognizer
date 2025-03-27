use crate::{binary_load::load_neuron, network};
use std::fs::File;

pub fn guess_answer(layers: &mut Vec<LiteLayer>, image: &Vec<f64>) {
    let layers_last_i = layers.len() - 1;
    let input_layer = layers.first_mut().unwrap();
    input_layer.set_neurons_activations(image, layers_last_i == 0);
    let mut iter_layers = layers.iter_mut();
    let mut shallower_layer = iter_layers.next().unwrap();
    let mut current_layer_i: usize = 1;
    for current_layer in iter_layers {
        current_layer.set_neurons_activations(
            &shallower_layer.get_neurons_activations(),
            current_layer_i == layers_last_i,
        );
        shallower_layer = current_layer;
        current_layer_i += 1;
    }
    //SOFTMAX
    let mut exp_sum = 0f64;
    for ref mut neuron in &mut layers.last_mut().unwrap().neurons {
        neuron.activation = f64::exp(neuron.activation);
        exp_sum += neuron.activation;
    }
    for ref mut neuron in &mut layers.last_mut().unwrap().neurons {
        neuron.activation /= exp_sum;
    }
}

pub struct LiteLayer {
    neurons: Vec<LiteNeuron>,
}

impl LiteLayer {
    pub fn new(file: &mut File, size_this_layer: usize, size_shallower_layer: usize) -> LiteLayer {
        let mut neurons: Vec<LiteNeuron> = Vec::with_capacity(size_this_layer);
        for _ in 0..size_this_layer {
            neurons.push(LiteNeuron::new(file, size_shallower_layer));
        }
        LiteLayer { neurons }
    }

    pub fn set_neurons_activations(
        &mut self,
        ref_shallower_activations: &Vec<f64>,
        is_output_layer: bool,
    ) {
        for neuron in &mut self.neurons {
            neuron.set_activation(ref_shallower_activations, is_output_layer);
        }
    }

    pub fn get_neurons_activations(&self) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| neuron.activation)
            .collect()
    }
}

struct LiteNeuron {
    weights: Vec<f64>,
    bias: f64,
    activation: f64,
}

impl LiteNeuron {
    pub fn new(file: &mut File, size_shallower_layer: usize) -> LiteNeuron {
        if let Ok((weights, bias)) = load_neuron(file, size_shallower_layer) {
            LiteNeuron {
                weights,
                bias,
                activation: 0f64,
            }
        } else {
            panic!("モデルの読み込み時に問題が発生しました。");
        }
    }

    pub fn set_activation(&mut self, ref_shallower_activations: &Vec<f64>, is_output_layer: bool) {
        self.activation = 0f64;
        for (weight, shallower_activation) in self.weights.iter().zip(ref_shallower_activations) {
            self.activation += weight * shallower_activation;
        }
        self.activation = if is_output_layer {
            self.activation + self.bias
        } else {
            network::leaky_relu(self.activation + self.bias)
        };
    }
}
