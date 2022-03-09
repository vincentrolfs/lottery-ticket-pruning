function arrayToTypedArray(array) {
    return Float32Array.from(array.flat(Infinity));
}

function typedArrayToArray(typedArray, dim0, dim1, dim2) {
    if (typedArray.length !== dim0 * dim1 * dim2) {
        throw "Assertion failed: Dimensions are lying";
    }

    const array = [];

    for (let i0 = 0; i0 < dim0; i0++) {
        for (let i1 = 0; i1 < dim1; i1++) {
            for (let i2 = 0; i2 < dim2; i2++) {
                if (array[i0] === undefined) {
                    array[i0] = [];
                }
                if (array[i0][i1] === undefined) {
                    array[i0][i1] = [];
                }

                array[i0][i1][i2] = typedArray[i0 * dim1 * dim2 + i1 * dim2 + i2];
            }
        }
    }

    return array;
}

function relu(input) {
    return input.map(i => Math.max(0, i))
}

function shortcut(youngInput, youngDim, oldInput, oldDim) {
    let shapeChange = false;

    if (youngInput.length !== youngDim[0] * youngDim[1] * youngDim[2]) {
        throw "Assertion failed: youngDim is lying";
    }

    if (oldInput.length !== oldDim[0] * oldDim[1] * oldDim[2]) {
        throw "Assertion failed: oldDim is lying";
    }

    if (youngDim[0] === oldDim[0]) {
        if (youngDim[1] !== oldDim[1] || youngDim[2] !== oldDim[2]) {
            throw "Assertion failed: Image resolutions of shortcut inputs differ, even though amount of channels is the same";
        }
    } else {
        if (youngDim[0] !== 2 * oldDim[0]) {
            throw "Assertion failed: Amount of channels of shortcut inputs do not differ by factor of 2, even though amount of channels is different";
        }

        if (2 * youngDim[1] !== oldDim[1] || 2 * youngDim[2] !== oldDim[2]) {
            throw "Assertion failed: Image resolutions of shortcut inputs do not differ by factor of 2, even though amount of channels is different";
        }

        shapeChange = true;
    }

    const oldMultiplier = shapeChange ? 2 : 1;
    const output = new Float32Array(youngInput);

    for (let youngChannel = 0; youngChannel < youngDim[0]; youngChannel++) {
        const oldChannel = shapeChange ? youngChannel - Math.floor(youngDim[0] / 4) : youngChannel;

        for (let x = 0; x < youngDim[1]; x++) {
            for (let y = 0; y < youngDim[2]; y++) {
                if (0 <= oldChannel && oldChannel < oldDim[0]) {
                    output[
                    youngChannel * youngDim[1] * youngDim[2]
                    + x * youngDim[2]
                    + y
                        ] += oldInput[
                    oldChannel * oldDim[1] * oldDim[2]
                    + oldMultiplier * x * oldDim[2]
                    + oldMultiplier * y
                        ]
                }
            }
        }
    }

    return output
}

function channelWiseMean(input, dim0, dim1, dim2) {
    if (input.length !== dim0 * dim1 * dim2) {
        throw "Assertion failed: Dimensions are lying";
    }

    const output = new Float32Array(dim0);

    for (let c = 0; c < dim0; c++) {
        output[c] = 0;

        for (let x = 0; x < dim1; x++) {
            for (let y = 0; y < dim2; y++) {
                output[c] += input[c * dim1 * dim2 + x * dim2 + y];
            }
        }

        output[c] /= dim1 * dim2;
    }

    return output
}

function argmax(input) {
    let max = -Infinity;
    let indexOfMax;

    for (let i = 0; i < input.length; i++) {
        if (input[i] > max) {
            max = input[i];
            indexOfMax = i;
        }
    }

    return indexOfMax;
}

const TRANSLATIONS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
];