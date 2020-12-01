using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.Layers;
using NeuralNetwork.LossClasses;
using Numpy;

namespace NeuralNetwork.Model
{
    public class Model
    {
        private List<ILayer> _layers;
        private CalculateEntropyLoss _loss;

        public Model()
        {
            this._layers = new List<ILayer>();
        }

        public void Add(ILayer layer) => this._layers.Add(layer);

        public void Set(CalculateEntropyLoss loss) => this._loss = loss;
        // seite 512 weiterschauen
    }
}
