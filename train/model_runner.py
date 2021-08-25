def predict(model, input_data):
    model.eval()

    outputs = model(input_data)
    _, output_indices = outputs[0].max(1)

    return output_indices, outputs


def predict_one(model, single_input):
    input_data = single_input[None, None, ...]
    return predict(model, input_data)


def predict_batch(model, input_batch):
    input_data = input_batch[:, None, ...]
    return predict(model, input_data)
