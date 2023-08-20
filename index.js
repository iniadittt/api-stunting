const express = require('express');
const tfjs = require('@tensorflow/tfjs-node');
const path = require('path');

const app = express();
const PORT = 3000;

app.use(express.json());

const loadModel = async() => {
    const model = await tfjs.loadLayersModel('file://' + path.join(__dirname, 'model/model.json'));
    return model;
}

let model;
loadModel().then(response => model = response).catch(error => console.log(error));

app.get('/', (req, res) => {
    return res.status(200).json({ status: 200, message: 'Hello World' });
})

app.post('/predict', async(req, res) => {
    const { sex, age, birthWeight, birthLength, bodyWeight, bodyLength, asiEksklusif } = req.body;
    const gender = sex === 'Laki-laki' ? 1 : sex === 'Perempuan' ? 0 : null;
    if (gender === null) return res.status(400).json({ status: 400, message: 'Sex tidak sesuai' });
    const asi = asiEksklusif === 'Yes' ? 1 : asiEksklusif === 'No' ? 0 : null;
    if (asi === null) return res.status(400).json({ status: 400, message: 'ASI Eksklusif tidak sesuai' });
    const temp = tfjs.tensor2d([
        [gender, age, birthWeight, birthLength, bodyWeight, bodyLength, asi]
    ]);
    const result = model.predict(temp);
    const presentase = result.dataSync()[0];
    return res.status(200).json({ status: 200, message: 'Predict berhasil!', data: { presentase, stunting: presentase > 0.5 ? true : false } });
})

app.listen(PORT, () => console.log(`Server running at port : ${PORT}`))