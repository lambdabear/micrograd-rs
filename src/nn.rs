use super::engine::Scalar;
use rand::Rng;
use thiserror::Error;

pub struct Neuron {
    w: Vec<Scalar<f32>>,
    b: Scalar<f32>,
    nonlin: bool,
}

impl Neuron {
    pub fn new(nin: usize, nonlin: bool) -> Self {
        let mut rng = rand::thread_rng();
        let mut w = vec![];

        for _ in 0..nin {
            let wi = rng.gen_range(-1.0..1.0);
            w.push(Scalar::new(wi, ""));
        }

        Self {
            w,
            b: Scalar::new(0.0, ""),
            nonlin,
        }
    }

    pub fn output(&mut self, input: Vec<Scalar<f32>>) -> Result<Scalar<f32>, NeuronError> {
        let mut output = Scalar::new(0.0, "");

        if self.w.len() != input.len() {
            return Err(NeuronError::InputLenErr);
        };

        for i in 0..self.w.len() {
            let xi = input[i].clone();
            let wi = self.w[i].clone();

            output += xi * wi;
        }

        output += self.b.clone();

        if self.nonlin {
            Ok(output.tanh())
        } else {
            Ok(output.clone())
        }
    }

    pub fn parameters(&self) -> Vec<Scalar<f32>> {
        let mut w = self.w.clone();
        w.push(self.b.clone());

        w
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonlin: bool) -> Self {
        let mut neurons = vec![];

        for _ in 0..nout {
            neurons.push(Neuron::new(nin, nonlin));
        }

        Self { neurons }
    }

    pub fn output(&mut self, input: Vec<Scalar<f32>>) -> Result<Vec<Scalar<f32>>, NeuronError> {
        let mut output = vec![];

        for neuron in &mut self.neurons {
            let o = neuron.output(input.clone())?;

            output.push(o);
        }

        Ok(output)
    }

    pub fn parameters(&self) -> Vec<Scalar<f32>> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: &[usize]) -> Self {
        let mut layers = vec![];

        if nouts.len() > 0 {
            layers.push(Layer::new(nin, nouts[0], 0 != nouts.len() - 1));

            if nouts.len() > 1 {
                for i in 0..nouts.len() - 1 {
                    layers.push(Layer::new(nouts[i], nouts[i + 1], i != nouts.len() - 2))
                }
            }
        }

        Self { layers }
    }

    pub fn output(&mut self, mut input: Vec<Scalar<f32>>) -> Result<Vec<Scalar<f32>>, NeuronError> {
        for layer in &mut self.layers {
            input = layer.output(input)?;
        }

        Ok(input)
    }

    pub fn parameters(&self) -> Vec<Scalar<f32>> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}

#[derive(Error, Debug)]
pub enum NeuronError {
    #[error("input data length error")]
    InputLenErr,
}
