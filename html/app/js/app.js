const ALL_SEEDS = [11696672, 15713537, 36569120, 70206358, 75504233, 83494940, 90478944, 92519636, 95284986, 96488735];
const ALL_COMPRESSIONS = [1, 2, 4, 8, 16, 32, 64];

const modelSelectElement = document.getElementById('select');
const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');
const statsDiv = document.getElementById('stats');
const detailsDiv = document.getElementById('details');

const uploadElement = document.getElementById('upload');
const uploadPreviewElement = document.getElementById('canvas');
const uploadResultElement = document.getElementById('uploadResult');

let isRunning = false;
let computedLabels = [];
let computationTimes = [];
let processedImages = 0;
let correctImages = 0;
let totalTime = 0;

function initExperiment() {
    computedLabels = [];
    computationTimes = [];
    processedImages = 0;
    totalTime = 0;
    correctImages = 0;

    _initialRender();
}

function startExperiment() {
    isRunning = true;
    startButton.disabled = true;
    stopButton.disabled = false;

    initExperiment();

    _markImage(0);
    setTimeout(_experimentStep(0, function () {
        isRunning = false;
        startButton.disabled = false;
        stopButton.disabled = true;
    }), 500);
}

function stopExperiment() {
    stopButton.disabled = true;
    isRunning = false;
}

function _experimentStep(i, onFinish) {
    return function () {
        _performMeasurement(i);
        _renderImage(i);

        if (isRunning && i < images.length - 1) {
            _markImage(i + 1);
            setTimeout(_experimentStep(i + 1, onFinish), 0);
        } else {
            onFinish();
        }
    }
}

function _performMeasurement(i) {
    const startTime = performance.now();

    computedLabels[i] = model(images[i]);

    const endTime = performance.now();
    const durationSeconds = (endTime - startTime) / 1000;

    computationTimes[i] = durationSeconds;
    totalTime += durationSeconds;
    processedImages++;
    correctImages += (computedLabels[i] === trueLabels[i]) ? 1 : 0;
}

function _initialRender() {
    _renderStats();

    let details = ""

    for (let i = 0; i < images.length; i++) {
        details += '<div><h3 id="heading' + i + '">Image ' + (i + 1) + ' <span>(computing...)</span></h3>';
        details += '<img src="images/' + i + '.png"/>';
        details += '<div id="content' + i + '"></div>';
        details += "</div>";
    }

    detailsDiv.innerHTML = details;
}

function _markImage(i) {
    document.getElementById('heading' + i).classList.add("loading");
}

function _renderImage(i) {
    _renderStats();

    const trueLabel = trueLabels[i];
    const computedLabel = computedLabels[i];
    const correct = (trueLabel === computedLabel);
    const correctClass = correct ? 'correct' : 'wrong';

    document.getElementById('heading' + i).classList.remove("loading");
    document.getElementById('heading' + i).classList.add(correctClass);

    let content = "<br/>True label: " + TRANSLATIONS[trueLabel];
    content += "<br/>Computed label: " + TRANSLATIONS[computedLabel];
    content += correct ? '<br/><span class="correct">Correct</span>' : '<br/><span class="wrong">Wrong</span>';
    content += '<br/><br/>Computation time: ' + computationTimes[i].toFixed(3) + 's';

    document.getElementById('content' + i).innerHTML = content;
}

function _renderStats() {
    statsDiv.innerHTML = "Processed: " + processedImages + " / " + images.length;
    statsDiv.innerHTML += "<br/>Correct predictions: " + correctImages + " / " + processedImages + " (" + (100 * correctImages / processedImages).toFixed(2) + "%)";
    statsDiv.innerHTML += "<br/>Total computation time: " + totalTime.toFixed(3) + "s"
    statsDiv.innerHTML += "<br/>Average time per image: " + (totalTime / processedImages).toFixed(3) + "s"
}

function addEventListeners() {
    startButton.addEventListener("click", startExperiment)
    stopButton.addEventListener("click", stopExperiment)
    modelSelectElement.addEventListener("change", function () {
        startButton.disabled = true;
        uploadElement.disabled = true;

        if (!modelSelectElement.value) {
            return;
        }

        const [compression, seed] = modelSelectElement.value.split("-");
        modelSelectElement.disabled = true;

        const scriptTag = document.createElement('script');
        scriptTag.onload = function () {
            modelSelectElement.disabled = false;
            startButton.disabled = false;
            uploadElement.disabled = false;
        }
        scriptTag.src = "../models_js/lt_traditional_" + seed + "_" + compression.padStart(2, "0") + ".000_round0" + (compression === "1" ? "0" : "1") + "_final.js";
        document.body.appendChild(scriptTag);
    })
    uploadElement.addEventListener("change", function () {
        function getImageData(img) {
            const ctx = uploadPreviewElement.getContext("2d");
            uploadPreviewElement.width = img.width;
            uploadPreviewElement.height = img.height;
            ctx.drawImage(img, 0, 0);

            return ctx.getImageData(0, 0, uploadPreviewElement.width, uploadPreviewElement.height);
        }

        function getImageChannels(imageData) {
            const MEANS = [0.485, 0.456, 0.406];
            const STDS = [0.229, 0.224, 0.225];
            const channels = [[[]], [[]], [[]]];

            imageData.data.forEach((value, i) => {
                if (i % 4 === 3) {
                    return;
                }

                const c = i % 4;
                const channel = channels[c];
                let row = channel[channel.length - 1];

                if (row.length >= imageData.width) {
                    channel.push([]);
                    row = channel[channel.length - 1];
                }

                row.push((value / 255 - MEANS[c]) / STDS[c]);
            });

            return arrayToTypedArray(channels);
        }

        function analyzeImage(img) {
            uploadPreviewElement.width = uploadPreviewElement.height = 0;
            uploadResultElement.innerHTML = "Loading..."

            if (img.width !== 32 || img.height !== 32) {
                uploadResultElement.innerHTML = "Image must be 32x32 pixel"
                return
            }

            const data = getImageData(img);

            if (!model) {
                return
            }

            const channels = getImageChannels(data);

            window.setTimeout(function () {
                uploadResultElement.innerHTML = TRANSLATIONS[model(channels)];
            }, 0)
        }

        if (!uploadElement.files[0]) {
            return;
        }

        const reader = new FileReader();

        reader.addEventListener("load", function (event) {
            const img = new Image();
            img.onload = function () {
                analyzeImage(img);
            }
            img.src = event.target.result;
        });
        reader.readAsDataURL(uploadElement.files[0]);
    })
}

function createCompressionOptions() {
    for (let i = 0; i < ALL_SEEDS.length; i++) {
        const seed = ALL_SEEDS[i];

        for (const compression of ALL_COMPRESSIONS) {
            modelSelectElement.innerHTML += `<option value="${compression}-${seed}">${compression}x Compression - Seed ${i + 1}</option>`;
        }
    }
}


function run() {
    createCompressionOptions();
    addEventListeners();
    initExperiment();
}

run();