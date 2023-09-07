#[derive(Copy, Clone)]
struct Layer<const PRECEDING_NEURONS: usize, const NEURONS: usize> {
    values: [f32; NEURONS],
    weights: [[f32; PRECEDING_NEURONS]; NEURONS],
    biases: [f32; NEURONS]
}

impl<const PRECEDING_NEURONS: usize, const NEURONS: usize> Layer<PRECEDING_NEURONS, NEURONS> {
    fn empty() -> Self {
        Layer {
            values: [0f32; NEURONS],
            weights: [[1f32; PRECEDING_NEURONS]; NEURONS],
            biases: [0f32; NEURONS]
        }
    }
    fn pass(&mut self, inputs: [f32; PRECEDING_NEURONS]) {
        for neuron_index in 0..NEURONS {
            self.values[neuron_index] = self.biases[neuron_index];
            for input_index in 0..PRECEDING_NEURONS {
                self.values[neuron_index] += inputs[input_index] * self.weights[neuron_index][input_index];
            }
        }
    }
    fn relu_activation(&mut self) {
        for index in 0..NEURONS {
            self.values[index] =  self.values[index].max(0f32);
        }
    }
    fn sigmoid_activation(&mut self) {
        for index in 0..NEURONS {
            self.values[index] =  1f32 / (1f32 + (-self.values[index]).exp());
        }
    }
}

// Note that BONUS_LAYERS is the number of hidden layers IN ADDITION to the default hidden first layer and output layer. So a network with BONUS_LAYERS=2 will have 3 hidden layers plus the output layer
#[derive(Copy, Clone)]
struct Brain<const INPUTS: usize, const BONUS_LAYERS: usize, const HIDDEN_LAYER_WIDTH: usize, const OUTPUTS: usize> {
    first_layer: Layer<INPUTS, HIDDEN_LAYER_WIDTH>,
    bonus_layers: [Layer<HIDDEN_LAYER_WIDTH, HIDDEN_LAYER_WIDTH>; BONUS_LAYERS],
    output_layer: Layer<HIDDEN_LAYER_WIDTH, OUTPUTS>
}

impl<const INPUTS: usize, const BONUS_LAYERS: usize, const HIDDEN_LAYER_WIDTH: usize, const OUTPUTS: usize> Brain<INPUTS, BONUS_LAYERS, HIDDEN_LAYER_WIDTH, OUTPUTS> {
    fn new() -> Self {
        Brain {
            first_layer: Layer::empty(),
            bonus_layers: [Layer::empty(); BONUS_LAYERS],
            output_layer: Layer::empty()
        }
    }
    fn think(&mut self, inputs: [f32; INPUTS]) {
        self.first_layer.pass(inputs);
        self.first_layer.relu_activation();
        if BONUS_LAYERS == 0 {
            return
        }
        self.bonus_layers[0].pass(self.first_layer.values);
        self.bonus_layers[0].relu_activation();
       
        for index in 1..BONUS_LAYERS {
            self.bonus_layers[index].pass(self.bonus_layers[index - 1].values);
            self.bonus_layers[index].relu_activation();
        }

        self.output_layer.pass(self.bonus_layers[BONUS_LAYERS - 1].values);
        self.output_layer.sigmoid_activation();
    }
    fn print(&self) {
        for value in self.first_layer.values {
            print!("{}, ", value);
        }
        println!();
        for layer in &self.bonus_layers {
            for value in layer.values {
                print!("{}, ", value);
            }
            println!();
        }
        for value in self.output_layer.values {
            print!("{}, ", value);
        }
        println!();
    }
}


fn main() {
    let mut b: Brain<1, 1, 2, 4> = Brain::new();
    b.think([1f32; 1]);
    b.print();
}
