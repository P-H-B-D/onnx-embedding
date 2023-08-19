const ort = require('onnxruntime-node');

function normalize(v) {
    if (v.length === 0 || v[0].length === 0) return [];
    
    const norms = v.map(vec => 
        Math.sqrt(vec.reduce((acc, val) => acc + val * val, 0)) || 1e-12
    );
    
    const result = new Float32Array(v.length * v[0].length);
    for (let i = 0; i < v.length; i++) {
        for (let j = 0; j < v[0].length; j++) {
            result[i * v[0].length + j] = v[i][j] / norms[i];
        }
    }

    return result;
}

function meanPoolingWithAttentionWeighting(lastHiddenState, attentionMask) {
    const sentenceCount = lastHiddenState.dims[0];
    const tokenCount = lastHiddenState.dims[1];
    const hiddenSize = lastHiddenState.dims[2];
    const lastHiddenStateData = lastHiddenState.data;
    const attentionMaskData = attentionMask.data;

    const embeddings = new Array(sentenceCount);
    for (let i = 0; i < sentenceCount; i++) {
        const embedding = new Float32Array(hiddenSize).fill(0);
        let sumAttention = 0;

        for (let j = 0; j < tokenCount; j++) {
            const attentionValue = Number(attentionMaskData[i * tokenCount + j]);
            sumAttention += attentionValue;

            for (let k = 0; k < hiddenSize; k++) {
                embedding[k] += lastHiddenStateData[i * tokenCount * hiddenSize + j * hiddenSize + k] * attentionValue;
            }
        }

        if (sumAttention === 0) sumAttention = 1e-9;
        for (let k = 0; k < hiddenSize; k++) {
            embedding[k] /= sumAttention;
        }

        embeddings[i] = embedding;
    }

    return normalize(embeddings);
}

function getRandomInt(max) {
    return BigInt(Math.floor(Math.random() * max));
}

function generateTensor(data, dataType, dims) {
    return new ort.Tensor(dataType, data, dims);
}

async function cvtToTokens(inputString) {
    const { AutoTokenizer } = await import('@xenova/transformers');
    let tokenizer = await AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2');
    let { input_ids } = await tokenizer(inputString);
    const dataBigIntArray = Array.from(input_ids.data, value => BigInt(value));
    paddingNum = 256 - dataBigIntArray.length;
    let input_ids_return = dataBigIntArray.concat(new Array(paddingNum).fill(BigInt(0)));
    let attentionMask = new Array(dataBigIntArray.length).fill(BigInt(1)).concat(new Array(paddingNum).fill(BigInt(0)));
    return [input_ids_return, attentionMask];
}


async function main(inputString) {
    const session = await ort.InferenceSession.create('./onnx/model.onnx');
    console.log(await cvtToTokens("Hello World how are you"));
    
    
    // const input_ids= tokenize(inputString);

    const randomNumbers = [
        BigInt(101),
        ...Array.from({ length: 3 }, () => getRandomInt(2001)), //Replace with actual token ids.
        BigInt(102)
    ];

    const zeroEntries = new Array(251).fill(BigInt(0)); //Padding
    const input_ids = randomNumbers.concat(zeroEntries);


    
    const input_ids_Tensor = generateTensor(input_ids, 'int64', [1, 256]);

    const aMask = new Array(256).fill(BigInt(0)).fill(BigInt(1), 0, 5);
    const attention_mask_Tensor = generateTensor(input_ids, 'int64', [1, 256]);

    const token_type_ids = new Array(256).fill(BigInt(0));
    const token_type_ids_Tensor = generateTensor(token_type_ids, 'int64', [1, 256]);

    const inputName = session.inputNames;
    const outputName = session.outputNames;

    
    const feeds = {
        input_ids: input_ids_Tensor,
        attention_mask: attention_mask_Tensor,
        token_type_ids: token_type_ids_Tensor
    };
    
    const results = await session.run(feeds);
    const lastHiddenState = results.last_hidden_state;
    const embeddings = meanPoolingWithAttentionWeighting(lastHiddenState, attention_mask_Tensor);
    // console.log(embeddings)
    return embeddings;
}

main();
