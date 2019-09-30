import tensorflow as tf
from datetime import datetime

class TensorflowLogger:
    def __init__(self, log_dir):
        # initialize the logger
        self.timestamp = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
        self.writer = tf.summary.FileWriter(log_dir + '/summary_' + self.timestamp)
        self.variable_steps = {}

    def scalar(self, name, value, step=None):
        # get the step value for the current scalar
        if step is None:
            if name in self.variable_steps:
                step = self.variable_steps[name] = self.variable_steps[name] + 1
            else:
                step = self.variable_steps[name] = 0
        else:
            self.variable_steps[name] = step

        # create the summary and add it to the file writer
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
        self.writer.add_summary(summary, step)

    def flush(self):
        # flush the file writer
        self.writer.flush()
