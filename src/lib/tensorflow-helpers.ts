import * as tf from '@tensorflow/tfjs';

const MNIST_IMAGES_SPRITE_PATH =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';
const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const NUM_TRAIN_ELEMENTS = 55000;

class MnistData {
  private datasetImages: Float32Array | null = null;
  private datasetLabels: Uint8Array | null = null;
  private trainImages: Float32Array | null = null;
  private testImages: Float32Array | null = null;
  private trainLabels: Uint8Array | null = null;
  private testLabels: Uint8Array | null = null;

  async load() {
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const imgRequest = new Promise<void>((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        const datasetBytesBuffer =
          new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
            IMAGE_SIZE * chunkSize);
          ctx!.drawImage(
            img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
            chunkSize);

          const imageData = ctx!.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j++) {
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);

        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [imgResponse, labelsResponse] =
      await Promise.all([imgRequest, labelsRequest]);

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    this.trainImages =
      this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels =
      this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    this.testLabels =
      this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  }

  nextTrainBatch(batchSize: number) {
    return this.nextBatch(
      batchSize, [this.trainImages, this.trainLabels], () => {
        return Math.floor(Math.random() * this.trainImages!.length / IMAGE_SIZE);
      });
  }

  nextTestBatch(batchSize: number) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      return Math.floor(Math.random() * this.testImages!.length / IMAGE_SIZE);
    });
  }

  private nextBatch(batchSize: number, data: [Float32Array | null, Uint8Array | null], index: () => number) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = index();
      const image = data[0]!.slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
      batchImagesArray.set(image, i * IMAGE_SIZE);
      const label = data[1]!.slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]).reshape([batchSize, 28, 28, 1]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return { images: xs, labels };
  }
}

let mnistDataInstance: MnistData | null = null;

export async function loadMnistData() {
    if (!mnistDataInstance) {
        mnistDataInstance = new MnistData();
        await mnistDataInstance.load();
    }
    return mnistDataInstance;
}
