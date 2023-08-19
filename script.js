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

async function main(inputString) {
    const session = await ort.InferenceSession.create('./onnx/model.onnx');
    const randomNumbers = [
        BigInt(101),
        ...Array.from({ length: 3 }, () => getRandomInt(2001)), //Replace with actual token ids.
        BigInt(102)
    ];

    const zeroEntries = new Array(251).fill(BigInt(0)); //Padding
    const input_ids = randomNumbers.concat(zeroEntries);
    const input_ids_Tensor = generateTensor(input_ids, 'int64', [1, 256]);

    const aMask = new Array(256).fill(BigInt(0)).fill(BigInt(1), 0, 5);
    const attention_mask_Tensor = generateTensor(aMask, 'int64', [1, 256]);

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
    console.log(embeddings)
    return embeddings;
}

main();


// async function main(){
//     const fs = require('fs');
//     const Tokenizer = require('tokenizer');

//     // Load the tokenizer from a JSON file
//     const tokenizerConfig = JSON.parse(fs.readFileSync('./onnx/tokenizer.json', 'utf-8'));
//     // You may need to adjust the way you load the tokenizer configuration

//     // Initialize the tokenizer
//     const tokenizer = new Tokenizer(tokenizerConfig);

//     // Enable truncation (tokenizer does not provide explicit truncation, so you'll need to truncate the tokens manually)
//     const max_length = 256;

//     // Example input text
//     const inputText = "This is an example sentence to tokenize.";

//     // Tokenize the input text
//     const tokens = tokenizer.tokenize(inputText);

//     // Truncate the tokens to the maximum length
//     const truncatedTokens = tokens.slice(0, max_length);

//     console.log(truncatedTokens);
// }

// main();



// const MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2";

// function normalize(v) {
//     const norm = v.map(row => Math.sqrt(row.reduce((sum, val) => sum + val * val, 0)));
//     for (let i = 0; i < v.length; i++) {
//         norm[i] = norm[i] === 0 ? 1e-12 : norm[i];
//         v[i] = v[i].map(val => val / norm[i]);
//     }
//     return v;
// }

// async function getEmbeddings(documents, batch_size = 32) {

//     let { AutoTokenizer, PreTrainedTokenizer } = await import('@xenova/transformers');
//     let tokenizerJSON=JSON.parse(fs.readFileSync('./onnx/tokenizer.json', 'utf8'));
//     let tokenizerConfig=JSON.parse(fs.readFileSync('./onnx/tokenizer_config.json', 'utf8'));
//     let tokenizerPT = new PreTrainedTokenizer(tokenizerJSON, tokenizerConfig);
//     let { input_ids } = await tokenizerPT('I love transformers!');
//     console.log(input_ids.data);
    
//     const maxSeqLength = 256;
//     const session = await ort.InferenceSession.create('./onnx/model.onnx');
//     const inputName = session.inputNames;
//     const outputName = session.outputNames;
//     console.log(`input name : ${inputName}`);
//     console.log(`output name : ${outputName}`);

//     let allEmbeddings = [];
//     for (let i = 0; i < documents.length; i += batch_size) {
//         let batch = documents.slice(i, i + batch_size);
//         let input_ids = batch.map(d => tokenizerPT(d));

//         let input_ids_Tensor=new ort.Tensor('float32', input_ids, [input_ids.length, maxSeqLength]);

//         let attention_mask = Array.from(input_ids).map(e => Array.from(e).map(id => (id > 0) ? 1 : 0));

//         let attention_mask_Tensor=new ort.Tensor('float32', attention_mask, [attention_mask.length, maxSeqLength]);
        
//         // Assume token_type_ids are all zeros as in the provided example
//         let token_type_ids = Array.from(input_ids).map(e => Array.from(e).map(() => 0));

//         let token_type_ids_Tensor=new ort.Tensor('float32', token_type_ids, [token_type_ids.length, maxSeqLength]);


//         const feeds = { input_ids: input_ids_Tensor,
//             attention_mask: attention_mask_Tensor,
//             token_type_ids: token_type_ids_Tensor };

  
//         const results = await session.run(feeds);
//         let modelOutput=results;
//         let last_hidden_state = modelOutput[0];

//         let embeddings = []; // Placeholder logic for processing the output from the ONNX model

//         allEmbeddings.push(embeddings);
//     }

//     return allEmbeddings.flat();

// }

// async function dub(){
//     const sampleText = "This is a sample text that is likely to overflow";  // Shortened for brevity
//     const embeddings = await getEmbeddings([sampleText, sampleText]);
//     console.log(embeddings);
// }
// dub();



// class DefaultEmbeddingModel {
//     constructor() {
//         this.tokenizer = new natural.WordTokenizer();
//         this.session = onnx.InferenceSession.create("onnx/model.onnx");
//         const inputName = this.session.inputNames;
//         const outputName = this.session.outputNames;
//         console.log(`input name : ${inputName}`);
//         console.log(`output name : ${outputName}`);
//         this.maxSeqLength = 256;
//     }

//     async __call__(documents, batch_size = 32) {
//         let all_embeddings = [];
//         for (let i = 0; i < documents.length; i += batch_size) {
//             let batch = documents.slice(i, i + batch_size);
//             let encoded = batch.map(d => this.tokenizer.tokenize(d));
//             let input_ids = encoded.map(e => e.slice(0, this.maxSeqLength).map(token => (token in this.tokenizer ? this.tokenizer[token] : 0)));
//             let attention_mask = input_ids.map(e => e.map(token => (token > 0 ? 1 : 0)));
            
//             let onnx_input = {
//                 input_ids: input_ids,
//                 attention_mask: attention_mask,
//                 token_type_ids: input_ids.map(e => new Array(e.length).fill(0))
//             };
            
//             let model_output = await this.session.run(onnx_input);
//             let last_hidden_state = model_output.values().next().value.data;
//             let input_mask_expanded = attention_mask.map(mask => mask.map(m => new Array(last_hidden_state[0].length).fill(m)));
            
//             let embeddings = [];
//             for (let j = 0; j < last_hidden_state.length; j++) {
//                 let sum = new Array(last_hidden_state[j].length).fill(0);
//                 for (let k = 0; k < last_hidden_state[j].length; k++) {
//                     sum[k] = last_hidden_state[j][k] * input_mask_expanded[j][k];
//                 }
//                 let sum_mask = input_mask_expanded[j].reduce((acc, val) => acc + val[0], 0);
//                 sum_mask = Math.max(sum_mask, 1e-9);
//                 embeddings.push(sum.map(val => val / sum_mask));
//             }
            
//             all_embeddings.push(normalize(embeddings));
//         }
        
//         return all_embeddings.flat();
//     }
// }

// const sample_text = "This is a sample text that is likely to overflow the entire model and will be truncated. \
//     Keep writing and writing until we reach the end of the model.This is a sample text that is likely to overflow the entire model and \
//     will be truncated. Keep writing and writing until we reach the end of the model.This is a sample text that is likely to overflow the entire \
//     model and will be truncated. Keep writing and writing until we reach the end of the model. This is a sample text that is likely to overflow \
//     the entire model and will be truncated. Keep writing and writing until we reach the end of the model. This is a sample text that is likely to overflow  \
//     the entire model and will be truncated. Keep writing and writing until we reach the end of the model.";

// const model = new DefaultEmbeddingModel();

// model.__call__([sample_text, sample_text])
//     .then(embeddings => {
//         console.log(embeddings);
//     })
//     .catch(err => {
//         console.error("Error:", err);
//     });