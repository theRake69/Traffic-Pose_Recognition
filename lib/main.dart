import 'dart:io';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Pose Detection'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late Interpreter _interpreter;
  late List<String> _labels; // Store labels loaded from labels.txt
  List<String> _topLabels = []; // Store top five labels
  String? _imagePath; // Path to currently selected image

  @override
  void initState() {
    super.initState();
    loadModel();
    loadLabels(); // Load labels when initializing
  }

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/best32.tflite');
      print('Model loaded successfully');
      detectPose(); // After model is loaded, perform inference
    } catch (e) {
      print('Failed to load model: $e');
    }
  }

  Future<void> loadLabels() async {
    try {
      String labelsContent = await rootBundle.loadString('assets/labels.txt');
      _labels = labelsContent.split('\n').map((label) => label.trim()).toList();
      print('Labels loaded successfully: $_labels');
    } catch (e) {
      print('Failed to load labels: $e');
      _labels = []; // Initialize with empty list if loading fails
    }
  }

  Future<Uint8List> loadImage(String imagePath) async {
    try {
      File imageFile = File(imagePath);
      Uint8List bytes = await imageFile.readAsBytes();
      return bytes;
    } catch (e) {
      print('Error loading image: $e');
      throw Exception('Failed to load image.');
    }
  }


  Uint8List preprocessImage(Uint8List inputImage) {
    img.Image? image = img.decodeImage(inputImage);
    if (image == null) {
      throw Exception('Failed to decode image.');
    }

    // Resize the image to 224x224
    img.Image resizedImage = img.copyResize(image, width: 224, height: 224);

    // Convert image to Uint8List buffer
    var buffer = resizedImage.getBytes();

    return buffer;
  }

  Future<void> detectPose() async {
    if (_imagePath == null) {
      print('No image selected.');
      return;
    }

    Uint8List input = await loadImage(_imagePath!);
    Uint8List preprocessedInput = preprocessImage(input);

    // Prepare input and output buffers for TensorFlow Lite interpreter
    var inputBuffer = Float32List.fromList(preprocessedInput.map((e) => e / 255.0).toList());
    var output = List.generate(1, (index) =>
        List.generate(46, (index) =>
            List.generate(1029, (index) => 0.0)
        )
    );

    try {
      _interpreter.run(inputBuffer.buffer.asUint8List(), output);
      List<String> labels = extractLabels(output);
      print('Detected Labels: $labels');

      // Get top five labels based on confidence
      _topLabels = labels.sublist(0, labels.length > 5 ? 5 : labels.length);
      setState(() {}); // Update UI with top labels
    } catch (e) {
      print('Error running model: $e');
    }
  }

  List<String> extractLabels(List<List<List<double>>> output) {
    List<String> labels = [];

    try {
      for (int i = 0; i < output[0].length; i++) {
        double x = output[0][i][0];
        double y = output[0][i][1];
        double confidence = output[0][i][2];

        if (i < _labels.length) {
          String label = '${_labels[i]}: (${x.toStringAsFixed(2)}, ${y.toStringAsFixed(2)}) Confidence: ${confidence.toStringAsFixed(2)}';
          labels.add(label);
        } else {
          // labels.add('Label not found for index $i');
        }
      }

      labels.sort((a, b) {
        double confidenceA = double.parse(a.split('Confidence: ')[1]);
        double confidenceB = double.parse(b.split('Confidence: ')[1]);
        return confidenceB.compareTo(confidenceA);
      });

      labels = labels.sublist(0, labels.length > 5 ? 5 : labels.length);
    } catch (e) {
      print('Error extracting labels: $e');
    }

    return labels;
  }

  void _pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _imagePath = pickedFile.path;
        _topLabels.clear(); // Clear previous labels when a new image is selected
      });
      detectPose(); // Detect pose on the newly selected image
    }
  }


  @override
  void dispose() {
    _interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            ElevatedButton(
              onPressed: _pickImage,
              child: const Text('Pick Image'),
            ),
            const SizedBox(height: 20),
            const Text(
              'Pose Detection',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 20),
            _imagePath != null
                ? Column(
              children: [
                Image.file(
                  File(_imagePath!),
                  height: 200,
                  width: 200,
                  fit: BoxFit.cover,
                ),
                const SizedBox(height: 20),
                const Text(
                  'Top 5 Labels:',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                SizedBox(
                  height: 200,
                  child: ListView(
                    children: _topLabels.map((label) => Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Text(
                        label,
                        style: const TextStyle(fontSize: 16),
                      ),
                    )).toList(),
                  ),
                ),
              ],
            )
                : Container(),
          ],
        ),
      ),
    );
  }
}
