# Glossary of Terms

## Radio & Signal Processing

**Amplitude Modulation (AM)**
Modulation technique where signal information is encoded in the amplitude (height) of the carrier wave.

**Bandwidth**
The range of frequencies a signal occupies. Wider bandwidth allows faster data transmission but requires more spectrum.

**Bearing**
The direction from an observer to a target, typically measured in degrees (0-360°) from magnetic north.

**Decibel (dB)**
Logarithmic unit for expressing ratios of signal power or intensity. dBm = dB relative to 1 milliwatt.

**Demodulation**
Process of extracting the original information signal from a modulated carrier wave.

**Frequency Modulation (FM)**
Modulation technique where signal information is encoded in the frequency deviation of the carrier wave.

**I/Q Signal** (In-phase/Quadrature)
Representation of a modulated signal as two components at 90° phase difference. Fundamental to SDR processing.

**Mel-Spectrogram**
Time-frequency representation of a signal using perceptually-motivated (mel) frequency scale, commonly used in ML models.

**Multilateration**
Localization technique using time-of-arrival or bearing measurements from multiple stations to triangulate a source position.

**Phase**
Position in the waveform cycle, typically measured in degrees (0-360°).

**Receiver**
Equipment that captures and processes radio signals.

**Signal-to-Noise Ratio (SNR)**
Measure of signal strength relative to background noise, typically in dB.

**Triangulation**
Geometric method using measurements from 3+ reference points to determine an unknown position.

**WebSDR**
Software-defined radio receiver accessible over the internet via web browser, typically operated by enthusiasts.

---

## Machine Learning Terms

**CNN (Convolutional Neural Network)**
Deep learning architecture using convolutional layers, effective for image-like data (spectrograms).

**Epoch**
One complete pass through entire training dataset.

**Feature**
Input variable or derived measurement used by ML model. Example: mel-spectrogram frequency bins.

**Gradient Descent**
Optimization algorithm that iteratively improves model by adjusting weights in direction of decreasing loss.

**Hyperparameter**
Model configuration parameter set before training (e.g., learning rate, batch size).

**Inference**
Process of using a trained model to make predictions on new data.

**Loss Function**
Mathematical function measuring difference between predicted and actual values. Model optimization goal is to minimize.

**Model Registry**
Centralized repository storing trained models with versioning and metadata.

**ONNX (Open Neural Network Exchange)**
Open standard format for representing trained models, enabling cross-framework deployment.

**Precision/Recall**
Precision = true positives / all positives predicted. Recall = true positives / all actual positives.

**PyTorch Lightning**
High-level PyTorch framework simplifying training loops and distributed training.

**Quantization**
Technique reducing model size/inference time by using lower-precision numerical representations.

**Softmax**
Function converting model outputs to probability distribution over classes.

**Training**
Process of adjusting model weights using labeled data to minimize loss function.

**Validation Set**
Subset of data used during training to evaluate model performance and tune hyperparameters.

---

## Infrastructure Terms

**Broker**
Message intermediary (e.g., RabbitMQ) handling asynchronous communication between services.

**Cache**
Fast-access data store (Redis) reducing database queries and improving response time.

**Container**
Lightweight, isolated execution environment for applications (Docker).

**Database**
Persistent data storage system (PostgreSQL + TimescaleDB).

**ETL**
Extract-Transform-Load: process of data extraction, processing, transformation, and loading into storage.

**Hypertable**
TimescaleDB structure automatically partitioning time-series data for efficient storage and querying.

**Microservice**
Architectural approach decomposing application into small, independent services.

**Object Storage**
Scalable storage system for unstructured data (MinIO, AWS S3, Google Cloud Storage).

**Orchestration**
Automated management of containerized applications (Kubernetes).

**Queue**
FIFO (First-In-First-Out) data structure for task buffering and processing.

**Replica**
Copy of data or service for reliability and scaling purposes.

**Sidecar**
Auxiliary container running alongside main application container.

**Stateless**
Service design where each request is independent, enabling horizontal scaling.

**Time-Series Data**
Data points indexed in time order, such as sensor readings or signal samples.

---

## Project-Specific Terms

**Heimdall**
Project name: AI platform for radio source localization using distributed WebSDR network.

**Phase**
Major project milestone with specific objectives. Example: Phase 0 through Phase 10.

**RF Acquisition**
Process of collecting raw radio frequency signal data from WebSDR receivers.

**RF Task**
User-submitted job requesting RF acquisition from network of WebSDR stations.

**Localization**
Process of determining geographic position of a radio transmission source.

**Measurement**
Individual signal measurement from one WebSDR station for triangulation.

**Result**
Completed output of RF task containing estimated location and uncertainty.

**Uncertainty Quantification**
Process of estimating confidence bounds around predicted location.

**Session**
Work period during which progress is tracked and documented.

**Checkpoint**
Validated state reached after completing phase tasks.

---

## Statistics & Mathematics

**Confidence Interval**
Range of values within which true parameter likely lies with specified probability.

**Gaussian Distribution**
Probability distribution described by mean and standard deviation, basis for uncertainty quantification.

**Least Squares**
Optimization technique minimizing sum of squared errors between observed and predicted values.

**Maximum Likelihood**
Estimation method finding parameter values most likely to produce observed data.

**P-value**
Probability of observing test statistic if null hypothesis is true. Lower = more significant.

**Probability Distribution**
Mathematical function describing probability of different outcomes.

**Standard Deviation**
Measure of data spread around mean. ~68% of data within 1 SD, ~95% within 2 SD.

**Uncertainty**
Range of possible values for a measurement, typically expressed as ±value or confidence interval.

---

## API & Web Terms

**API (Application Programming Interface)**
Set of protocols and tools for building software applications.

**Endpoint**
Specific URL address for API resource or service.

**HTTP Status Code**
3-digit number indicating request result (200=OK, 404=Not Found, 500=Error).

**JSON (JavaScript Object Notation)**
Lightweight data exchange format using key-value pairs.

**Latency**
Time delay between request and response.

**Rate Limiting**
Restricting number of API requests per time period.

**REST (Representational State Transfer)**
Architectural style for web services using HTTP methods (GET, POST, etc.).

**WebSocket**
Full-duplex communication protocol enabling real-time bidirectional data exchange.

---

**Last Updated**: October 2025

**See Also**: [FAQ](./faqs.md) | [Documentation Index](./index.md)
