using System;
using System.Collections.Generic;
using System.Text;
using Numpy;

namespace NeuralNetwork.LossClasses
{
    public class CalculateEntropyLoss
    {
        public double CalcEntropyLoss(NDarray softmax_outputs, NDarray class_targets)
        {
            NDarray correct_confidences = null;

            var softmax_outputsClipped = np.clip(softmax_outputs
                                               , np.array(1e-7)
                                               , np.array(1 - 1e-7));

            var class_targetsClipped = np.clip(class_targets
                                             , np.array(1e-7)
                                             , np.array(1 - 1e-7));
            if (class_targets.len == 1)
            {
                correct_confidences = softmax_outputsClipped[softmax_outputs, class_targets];
            }
            else
            {
                correct_confidences = np.sum(softmax_outputsClipped * class_targetsClipped, 1);
            }

            var neg_log = -np.log(correct_confidences);

            var average_loss = np.mean(neg_log);
            return average_loss;
        }
    }
}
