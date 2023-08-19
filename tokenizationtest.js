export async function cvtToTokens(inputString) {
    const { AutoTokenizer } = await import('@xenova/transformers');
    let tokenizer = await AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2');
    let { input_ids } = await tokenizer(inputString);
    const dataBigIntArray = Array.from(input_ids.data, value => BigInt(value));
    paddingNum = 256 - dataBigIntArray.length;
    let input_ids_return = dataBigIntArray.concat(new Array(paddingNum).fill(BigInt(0)));
    let attentionMask = new Array(dataBigIntArray.length).fill(BigInt(1)).concat(new Array(paddingNum).fill(BigInt(0)));
    return [input_ids_return, attentionMask];
}