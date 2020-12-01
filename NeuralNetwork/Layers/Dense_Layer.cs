using System;
using System.Collections.Generic;
using System.Text;
using Numpy;

namespace NeuralNetwork.Layers
{
    public class Dense_Layer : ILayer
    {
        private NDarray _weights;
        private NDarray _biases;
        
        public Dense_Layer(int inputs, int neurons)
        {
            this._weights = 0.1 * np.random.randn(inputs, neurons);
            this._biases = np.zeros(( 1, neurons ));
        }

        public NDarray Forward(NDarray inputs)
        {
            //var neuron = new InputNeurons(inputs, _weights, _biases);
            return np.dot(inputs, _weights) + _biases; //neuron.Output();
        }
    }
}
