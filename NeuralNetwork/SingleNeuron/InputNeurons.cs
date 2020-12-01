using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using Numpy;

namespace NeuralNetwork.SingleNeuron
{
    public class InputNeurons
    {
        private NDarray _inputs;
        private NDarray _weights;
        private NDarray _biases;

        public InputNeurons(NDarray inputs, NDarray weights, NDarray biases)
        {
            this._inputs = inputs;
            this._weights = weights;
            this._biases = biases;
        }

        public NDarray Output()
        {
            return np.dot(_inputs, _weights.T) + _biases;
        }
    }
}
