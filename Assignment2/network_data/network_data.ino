// EE446 TinyML - Assignment 2, Question 5(c)
// Luke Valerio
// 5 test samples (X_test rows 0..4) from the seeded split (random_state=42).

#include <TensorFlowLite.h>
#include "network_model.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <math.h>

#define NUMBER_OF_INPUTS 38
#define NUMBER_OF_OUTPUTS 1
#define TENSOR_ARENA_SIZE (32 * 1024)

uint8_t tensor_arena[TENSOR_ARENA_SIZE];
tflite::ErrorReporter* error_reporter;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

const float X_test[5][38] = {
    -0.11024922321249885, -2.468724163053645, -0.9926628603286901, 
    0.7511112872365361, -0.007586440761970732, -0.004918644383724874, 
    -0.08948642202040107, -0.09507567152556495, -0.8092618187059747, 
    -0.011663642603760032, -0.036651869142258646, -0.024436507262009306, 
    -0.01238515036740332, -0.02618002418454278, -0.018609896340735923, 
    -0.04122119759327531, -0.0028174939213690777, -0.0975309439715147, 
    3.7280528465157605, 6.65324490222183, -0.6372092679572258, 
    -0.6319290328885425, -0.37436223991967527, -0.37443160310530493, 
    0.7712831058493207, -0.349683030873482, -0.37455970440553465, 
    0.7343425609306344, 0.3915636300872831, 0.21997736341067137, 
    -0.33321384013747096, 1.5263022423334498, -0.28910340026287856, 
    -0.6395319051152512, -0.6248707997445304, -0.38763462350750655, 
    -0.3763870260680415, -0.6563667617603728, -0.11024922321249885, 
    -0.12470615670462065, -0.44208308523109213, 0.7511112872365361, 
    -0.00772493434976162, -0.0048211623348653225, -0.08948642202040107, 
    -0.09507567152556495, 1.2356940323701657, -0.011663642603760032, 
    -0.036651869142258646, -0.024436507262009306, -0.01238515036740332, 
    -0.02618002418454278, -0.018609896340735923, -0.04122119759327531, 
    -0.0028174939213690777, -0.0975309439715147, -0.5249194340662569, 
    0.23765375772788483, -0.6372092679572258, -0.6319290328885425, 
    -0.37436223991967527, -0.37443160310530493, 0.7712831058493207, 
    -0.349683030873482, 0.04879489561066655, 0.7343425609306344, 
    1.2587542737799418, 1.0664013456654926, -0.43907816809041417, 
    -0.4801968475158174, -0.28910340026287856, -0.6395319051152512, 
    -0.6248707997445304, -0.38763462350750655, -0.3763870260680415, 
    0.6528228780141483, -0.11024922321249885, -0.12470615670462065, 
    -0.44208308523109213, 0.7511112872365361, -0.007726467489109859, 
    -0.004850755099697686, -0.08948642202040107, -0.09507567152556495, 
    1.2356940323701657, -0.011663642603760032, -0.036651869142258646, 
    -0.024436507262009306, -0.01238515036740332, -0.02618002418454278, 
    -0.018609896340735923, -0.04122119759327531, -0.0028174939213690777, 
    -0.0975309439715147, -0.6733804787682985, -0.28550603517076306, 
    -0.6372092679572258, -0.6319290328885425, -0.37436223991967527, 
    -0.37443160310530493, 0.7712831058493207, -0.349683030873482, 
    -0.37455970440553465, -1.7655107925814544, 1.2587542737799418, 
    1.0664013456654926, -0.43907816809041417, -0.027116407872434464, 
    0.155090881452997, -0.6395319051152512, -0.6248707997445304, 
    -0.38763462350750655, -0.3763870260680415, 0.6528228780141483, 
    -0.11024922321249885, -0.12470615670462065, 1.0873051789289023, 
    -0.7362346401101137, -0.00776224074056876, -0.004918644383724874, 
    -0.08948642202040107, -0.09507567152556495, -0.8092618187059747, 
    -0.011663642603760032, -0.036651869142258646, -0.024436507262009306, 
    -0.01238515036740332, -0.02618002418454278, -0.018609896340735923, 
    -0.04122119759327531, -0.0028174939213690777, -0.0975309439715147, 
    0.5230408814775668, -0.21666922031567779, 1.602663889932865, 
    1.6051037177847889, -0.37436223991967527, -0.37443160310530493, 
    -1.3214280114927377, -0.016929597707948767, -0.37455970440553465, 
    0.7343425609306344, -0.9363220430671004, -1.0496586099715604, 
    -0.12148518423158469, -0.4801968475158174, -0.28910340026287856, 
    1.6087590765792643, 1.6189552037455606, -0.38763462350750655, 
    -0.3763870260680415, 0.6528228780141483, -0.11024922321249885, 
    -0.12470615670462065, -1.6655936965590876, -0.7362346401101137, 
    -0.00776224074056876, -0.004918644383724874, -0.08948642202040107, 
    -0.09507567152556495, -0.8092618187059747, -0.011663642603760032, 
    -0.036651869142258646, -0.024436507262009306, -0.01238515036740332, 
    -0.02618002418454278, -0.018609896340735923, -0.04122119759327531, 
    -0.0028174939213690777, -0.0975309439715147, 1.4400061575784124, 
    -0.1340650424895755, 1.602663889932865, 1.6051037177847889, 
    -0.37436223991967527, -0.37443160310530493, -1.3441748714638466, 
    -0.016929597707948767, -0.37455970440553465, 0.7343425609306344, 
    -1.0266544017850858, -1.1387558712615415, -0.12148518423158469, 
    -0.4801968475158174, -0.28910340026287856, 1.6087590765792643, 
    1.6189552037455606, -0.38763462350750655, -0.3763870260680415, 
    -0.6563667617603728
};



const uint8_t y_test[5] = {
    0, 1, 1, 0, 0
};  // Actual labels for each sample

int8_t quantize_to_int8(float value, float scale, int zero_point) {
    int32_t quantized_value =
        static_cast<int32_t>(lroundf(value / scale)) + zero_point;

    if (quantized_value > 127) {
        quantized_value = 127;
    }

    if (quantized_value < -128) {
        quantized_value = -128;
    }

    return static_cast<int8_t>(quantized_value);
}

void setup() {
    Serial.begin(115200);
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    const tflite::Model* model = tflite::GetModel(network_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version does not match schema version.");
        return;
    }

    static tflite::MicroMutableOpResolver<10> micro_op_resolver;
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddLogistic();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate tensors!");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
        Serial.println("Expected INT8 input and output tensors.");
        return;
    }
}

void loop() {
    for (uint8_t i = 0; i < 5; i++) {
        // Load the i-th test sample data into the input tensor
        for (int j = 0; j < NUMBER_OF_INPUTS; j++) {
           input->data.int8[j] = quantize_to_int8(
            X_test[i][j],
            input->params.scale,
            input->params.zero_point
            );
        }

        // Run the model on this input and check for error
        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println("Failed to invoke!");
            continue;
        }

        // Question 5: Deploying the Quantized Model
        // (a) Implement code to obtain the sigmoid prediction from the INT8
        // output tensor and determine the predicted class label.
        // The output tensor is INT8, so it must be dequantized back to a float
        // probability using the output tensor's own scale and zero point:
        //     real_value = (quantized_value - zero_point) * scale
        // The result is the sigmoid output in [0, 1]; thresholding at 0.5 gives
        // the class (1 = normal, 0 = attack).
        float prediction = (output->data.int8[0] - output->params.zero_point) * output->params.scale;
        int predicted_class = (prediction > 0.5f) ? 1 : 0;

        // (b) Implement code to output Sample #, Predicted Class, and Actual Class for each sample to the serial monitor using Serial.print function.
        Serial.print("Sample #");
        Serial.print(i);
        Serial.print("  |  Predicted Class: ");
        Serial.print(predicted_class);
        Serial.print("  |  Actual Class: ");
        Serial.print(y_test[i]);
        Serial.print("  |  (sigmoid = ");
        Serial.print(prediction, 4);
        Serial.println(")");


        // Delay between predictions
        delay(1000);
    }

    // Delay before repeating the tests
    delay(10000);
}
