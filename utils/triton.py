
from typing import List
from urllib.parse import urlsplit

import numpy as np


class TritonRemoteModel:


    def __init__(self, url: str, endpoint: str = '', scheme: str = ''):
        if not endpoint and not scheme:  # Parse all args from URL string
            splits = urlsplit(url)
            endpoint = splits.path.strip('/').split('/')[0]
            scheme = splits.scheme
            url = splits.netloc

        self.endpoint = endpoint
        self.url = url

        # Choose the Triton client based on the communication scheme
        if scheme == 'http':
            import tritonclient.http as client  # noqa
            self.triton_client = client.InferenceServerClient(url=self.url, verbose=False, ssl=False)
            config = self.triton_client.get_model_config(endpoint)
        else:
            import tritonclient.grpc as client  # noqa
            self.triton_client = client.InferenceServerClient(url=self.url, verbose=False, ssl=False)
            config = self.triton_client.get_model_config(endpoint, as_json=True)['config']

        self.InferRequestedOutput = client.InferRequestedOutput
        self.InferInput = client.InferInput

        type_map = {'TYPE_FP32': np.float32, 'TYPE_FP16': np.float16, 'TYPE_UINT8': np.uint8}
        self.input_formats = [x['data_type'] for x in config['input']]
        self.np_input_formats = [type_map[x] for x in self.input_formats]
        self.input_names = [x['name'] for x in config['input']]
        self.output_names = [x['name'] for x in config['output']]

    def __call__(self, *inputs: np.ndarray) -> List[np.ndarray]:
        infer_inputs = []
        input_format = inputs[0].dtype
        for i, x in enumerate(inputs):
            if x.dtype != self.np_input_formats[i]:
                x = x.astype(self.np_input_formats[i])
            infer_input = self.InferInput(self.input_names[i], [*x.shape], self.input_formats[i].replace('TYPE_', ''))
            infer_input.set_data_from_numpy(x)
            infer_inputs.append(infer_input)

        infer_outputs = [self.InferRequestedOutput(output_name) for output_name in self.output_names]
        outputs = self.triton_client.infer(model_name=self.endpoint, inputs=infer_inputs, outputs=infer_outputs)

        return [outputs.as_numpy(output_name).astype(input_format) for output_name in self.output_names]
