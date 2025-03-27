use rand_distr::{Distribution, Normal};

pub fn guess_answer(layers: &mut Vec<Layer>, image: &Vec<f64>) {
    let layers_last_i = layers.len() - 1;
    let shallowest_layer = layers.first_mut().unwrap();
    shallowest_layer.set_neurons_activations(image, layers_last_i == 0);
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

pub fn backpropagation(
    layers: &mut Vec<Layer>,
    image: &Vec<f64>,
    learning_rate: f64,
    answer: usize,
) -> f64 {
    guess_answer(layers, image);
    let mut ideal_activations = [0f64; 10];
    ideal_activations[answer] = 1.0;
    for (i, neuron) in layers.last_mut().unwrap().neurons.iter_mut().enumerate() {
        neuron.delta = neuron.activation - ideal_activations.get(i).unwrap();
    }
    let mut rev_layers = layers.iter_mut().rev();
    let mut current_layer = rev_layers.next().unwrap();
    let mut is_output_layer = true;
    for shallower_layer in rev_layers {
        for ref mut neuron in &mut current_layer.neurons {
            neuron.stack_correction_activations(
                &mut shallower_layer.neurons,
                learning_rate,
                is_output_layer,
            );
        }
        current_layer = shallower_layer;
        is_output_layer = false;
    }
    for ref mut neuron in &mut current_layer.neurons {
        neuron.stack_correction_activations_shallowest_layer(image, learning_rate, is_output_layer);
    }
    -layers
        .last()
        .unwrap()
        .neurons
        .get(answer)
        .unwrap()
        .activation
        .ln()
}

pub fn apply_neurons_fixes(layers: &mut Vec<Layer>, size_batch: usize) {
    for ref mut layer in layers {
        for ref mut neuron in &mut layer.neurons {
            neuron.apply_fixes(size_batch);
        }
    }
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(
        size_this_layer: usize,
        size_shallower_layer: usize,
        is_output_layer: bool,
    ) -> Layer {
        let mut neurons: Vec<Neuron> = Vec::with_capacity(size_this_layer);
        for _ in 0..size_this_layer {
            neurons.push(if is_output_layer {
                Neuron::new_xavier(size_shallower_layer, size_this_layer)
            } else {
                Neuron::new_he(size_shallower_layer)
            });
        }
        Layer { neurons }
    }

    fn set_neurons_activations(
        &mut self,
        ref_shallower_activations: &Vec<f64>,
        is_output_layer: bool,
    ) {
        for neuron in &mut self.neurons {
            neuron.set_activation(ref_shallower_activations, is_output_layer);
        }
    }

    #[inline]
    pub fn get_neurons_activations(&self) -> Vec<f64> {
        self.neurons
            .iter()
            .map(|neuron| neuron.activation)
            .collect()
    }
}

pub struct Neuron {
    weights: Vec<f64>,
    fix_weights: Vec<f64>,
    bias: f64,
    fix_bias: f64,
    pre_activation: f64,
    activation: f64,
    delta: f64,
}

impl Neuron {
    #[inline]
    fn new_he(size_shallower_layer: usize) -> Neuron {
        let normal = Normal::new(0f64, (2.0 / size_shallower_layer as f64).sqrt()).unwrap();
        let mut rng = rand::thread_rng();
        Neuron {
            weights: (0..size_shallower_layer)
                .map(|_| normal.sample(&mut rng))
                .collect(),
            fix_weights: vec![0f64; size_shallower_layer],
            bias: 0f64,
            fix_bias: 0f64,
            pre_activation: 0f64,
            activation: 0f64,
            delta: 0f64,
        }
    }

    #[inline]
    fn new_xavier(size_shallower_layer: usize, size_this_layer: usize) -> Neuron {
        let normal = Normal::new(
            0f64,
            (2.0 / (size_this_layer + size_shallower_layer) as f64).sqrt(),
        )
        .unwrap();
        let mut rng = rand::thread_rng();
        Neuron {
            weights: (0..size_shallower_layer)
                .map(|_| normal.sample(&mut rng))
                .collect(),
            fix_weights: vec![0f64; size_shallower_layer],
            bias: 0f64,
            fix_bias: 0f64,
            pre_activation: 0f64,
            activation: 0f64,
            delta: 0f64,
        }
    }

    fn set_activation(&mut self, ref_shallower_activations: &Vec<f64>, is_output_layer: bool) {
        self.pre_activation = 0.0;
        for (weight, shallower_activation) in self.weights.iter().zip(ref_shallower_activations) {
            self.pre_activation += weight * shallower_activation;
        }
        self.pre_activation += self.bias;
        self.activation = if is_output_layer {
            self.pre_activation
        } else {
            leaky_relu(self.pre_activation)
        }
    }

    #[inline]
    fn stack_correction_activations(
        &mut self,
        ref_shallower_neurons: &mut Vec<Neuron>,
        learning_rate: f64,
        is_output_layer: bool,
    ) {
        let mut learning_amount = learning_rate * self.delta;
        if !is_output_layer {
            learning_amount *= derivative_leaky_relu(self.pre_activation);
        }
        for ((shallower_neuron, fix_weight), weight) in ref_shallower_neurons
            .iter_mut()
            .zip(self.fix_weights.iter_mut())
            .zip(self.weights.iter())
        {
            *fix_weight += learning_amount * shallower_neuron.activation;
            shallower_neuron.delta += self.delta * weight;
        }
        self.fix_bias += learning_amount;
        self.delta = 0f64;
    }

    fn stack_correction_activations_shallowest_layer(
        &mut self,
        image: &Vec<f64>,
        learning_rate: f64,
        is_output_layer: bool,
    ) {
        let mut learning_amount = learning_rate * self.delta;
        if !is_output_layer {
            learning_amount *= derivative_leaky_relu(self.pre_activation);
        }
        for (image_pixel, fix_weight) in image.iter().zip(self.fix_weights.iter_mut()) {
            *fix_weight += learning_amount * image_pixel;
        }
        self.fix_bias += learning_amount;
        self.delta = 0f64;
    }

    fn apply_fixes(&mut self, size_batch: usize) {
        let size_batch = size_batch as f64;
        for (weight, fix_waight) in self.weights.iter_mut().zip(&mut self.fix_weights) {
            *weight -= *fix_waight / size_batch;
            *fix_waight = 0f64;
        }
        self.bias -= self.fix_bias / size_batch;
        self.fix_bias = 0f64;
    }

    pub fn get_parameters(&self) -> (&Vec<f64>, f64) {
        (&self.weights, self.fix_bias)
    }
}

//LEAKY RELU
pub fn leaky_relu(x: f64) -> f64 {
    if x >= 0.0 { x } else { x * -0.04 }
}

pub fn derivative_leaky_relu(x: f64) -> f64 {
    if x >= 0.0 { 1.0 } else { -0.04 }
}
