def normalize_input(data):
    for i in range(len(data)):
        data[i] /= 255
    return data


def normalize_output(data):
    for i in range(len(data)):
        data[i] /= 99
    return data
