#include <stdio.h>
#include <stdint.h>
#include "test_images.c"
#include "calculations.c"
#include "weights.c"
#define batch_amt 1
#define img_input_channel 1
#define img_H 28
#define img_W 28
#define c_kernel_size 5
#define c_stride 1
#define maxpool_stride 2
#define maxpool_K 2

#define c1_output_channel 6
#define c1_padding 2
#define c1_H_in 28
#define c1_W_in 28
#define c2_output_channel 16
#define c2_padding 0
#define c3_output_channel 120
#define c3_padding 0
#define fc1_feature_out 84
#define fc2_feature_out 10

int main() {

    // IMAGE HAS BEEN NORMALIZED BEFORE SAVE
    const float quant_scale = 0.025556;
    const unsigned char quant_zero_point = 17;

    uint8_t quantized_img[batch_amt * img_input_channel * img_H * img_W];

    quantize_image(
        (const uint8_t *)image_1,
        batch_amt, img_input_channel,
        img_H, img_W,
        (uint8_t *)quantized_img,
        quant_scale, quant_zero_point
    );

    // UNCOMMENT TO VIEW QUANTIZED RESULT
    // printf("Quantized image:\n");
    // for (uint8_t batch = 0; batch < batch_amt; ++batch){
    //     for (uint8_t channel = 0; channel < img_input_channel; ++channel){
    //         for (uint8_t i = 0; i < img_H; ++i) {
    //             for (uint8_t j = 0; j < img_W; ++j) {
    //                 printf("%d,", quantized_img[((batch * img_input_channel + channel) * img_H + i) * img_W + j]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }


    const int c1_H_out = (img_H + 2*c1_padding - c_kernel_size)/c_stride + 1;
    const int c1_W_out = (img_W + 2*c1_padding - c_kernel_size)/c_stride + 1;
    uint8_t c1_output[batch_amt* c1_output_channel *c1_H_out * c1_W_out];
    conv2d(
        (uint8_t *)quantized_img,
        batch_amt, img_input_channel, img_H, img_W,
        (int8_t *)c1_weight,
        c1_output_channel, c_kernel_size,
        // &(c1_weight[0][0][0][0]), c1_output_channel, 5, 
        (uint8_t *)c1_output,
        // &(output[0][0][0][0]),
        (int32_t *)c1_bias,
        c1_real_scale,
        quant_zero_point, 0, c1_zero_point,
        c_stride, c1_padding
    );

    // UNCOMMENT TO VIEW C1 RESULT
    // printf("c1 result:\n");
    // for (uint8_t batch = 0; batch < batch_amt; ++batch){
    //     printf("batch %d\n", batch);
    //     for (uint8_t channel = 0; channel < c1_output_channel; ++channel){
    //         printf("channel %d\n",channel);
    //         for (uint8_t i = 0; i < c1_H_out; ++i) {
    //             // printf("i %d\n",i);
    //             for (uint8_t j = 0; j < c1_W_out; ++j) {
    //                 printf("%d,", c1_output[((batch * batch_amt + channel) * c1_H_out + i) * c1_W_out + j]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }
    
    uint8_t c1_maxpool_H_out = (c1_H_out - 2) / maxpool_stride + 1;
    uint8_t c1_maxpool_W_out = (c1_W_out - 2) / maxpool_stride + 1;
    uint8_t c1_maxpool_out[batch_amt* c1_output_channel* c1_maxpool_H_out* c1_maxpool_W_out];
    maxpool2d(
            (uint8_t *)c1_output,
            batch_amt, c1_output_channel, c1_H_out, c1_W_out,
            maxpool_K, maxpool_stride,
            (uint8_t *)c1_maxpool_out
        );
    // UNCOMMENT TO VIEW C1 MAXPOOL RESULT
    // printf("c1_maxpool result:\n");
    // for (uint8_t batch = 0; batch < batch_amt; ++batch){
    //     printf("batch %d\n", batch);
    //     for (uint8_t channel = 0; channel < c1_output_channel; ++channel){
    //         printf("channel %d\n",channel);
    //         for (uint8_t i = 0; i < c1_maxpool_H_out; ++i) {
    //             // printf("i %d\n",i);
    //             for (uint8_t j = 0; j < c1_maxpool_W_out; ++j) {
    //                 printf("%d,", c1_maxpool_out[((batch * batch_amt + channel) * c1_maxpool_H_out + i) * c1_maxpool_W_out + j]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }
    relu_quant(
        (uint8_t*) c1_maxpool_out,
        batch_amt, c1_output_channel, c1_maxpool_H_out, c1_maxpool_W_out,
        c1_zero_point
        );
    // UNCOMMENT TO VIEW C1 MAXPOOL RELU RESULT
    // printf("c1_maxpool_relu result:\n");
    // for (uint8_t batch = 0; batch < batch_amt; ++batch){
    //     printf("batch %d\n", batch);
    //     for (uint8_t channel = 0; channel < c1_output_channel; ++channel){
    //         printf("channel %d\n",channel);
    //         for (uint8_t i = 0; i < c1_maxpool_H_out; ++i) {
    //             // printf("i %d\n",i);
    //             for (uint8_t j = 0; j < c1_maxpool_W_out; ++j) {
    //                 printf("%d,", c1_maxpool_out[((batch * batch_amt + channel) * c1_maxpool_H_out + i) * c1_maxpool_W_out + j]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }

    // C2
    const int c2_H_out = (c1_maxpool_H_out + 2*c2_padding - c_kernel_size)/c_stride + 1;
    const int c2_W_out = (c1_maxpool_W_out + 2*c2_padding - c_kernel_size)/c_stride + 1;
    uint8_t c2_output[batch_amt* c2_output_channel* c2_H_out* c2_W_out];
    conv2d(
        (uint8_t *)c1_maxpool_out,
        batch_amt, c1_output_channel, c1_maxpool_H_out, c1_maxpool_W_out,
        (int8_t *)c2_weight,
        c2_output_channel, c_kernel_size,
        (uint8_t *)c2_output,
        (int32_t *)c2_bias,
        c2_real_scale,
        c1_zero_point, 0, c2_zero_point,
        c_stride, c2_padding
    );
    // printf("c2 result:\n");
    // for (uint8_t batch = 0; batch < batch_amt; ++batch){
    //     printf("batch %d\n", batch);
    //     for (uint8_t channel = 0; channel < c2_output_channel; ++channel){
    //         printf("channel %d\n",channel);
    //         for (uint8_t i = 0; i < c2_H_out; ++i) {
    //             // printf("i %d\n",i);
    //             for (uint8_t j = 0; j < c2_W_out; ++j) {
    //                 printf("%d,", c2_output[((batch * batch_amt + channel) * c2_H_out + i) * c2_W_out + j]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }

    // C2 MAXPOOLING
    uint8_t c2_maxpool_H_out = (c2_H_out - 2) / maxpool_stride + 1;
    uint8_t c2_maxpool_W_out = (c2_W_out - 2) / maxpool_stride + 1;
    uint8_t c2_maxpool_out[batch_amt* c2_output_channel* c2_maxpool_H_out* c2_maxpool_W_out];
    maxpool2d(
            (uint8_t *)c2_output,
            batch_amt, c2_output_channel, c2_H_out, c2_W_out,
            maxpool_K, maxpool_stride,
            (uint8_t *)c2_maxpool_out
        );
    relu_quant(
        (uint8_t*) c2_maxpool_out,
        batch_amt, c2_output_channel, c2_maxpool_H_out, c2_maxpool_W_out,
        c2_zero_point
        );

    // UNCOMMENT TO VIEW C2 MAXPOOL RELU RESULT
    // printf("c2_maxpool_relu result:\n");
    // for (uint8_t batch = 0; batch < batch_amt; ++batch){
    //     printf("batch %d\n", batch);
    //     for (uint8_t channel = 0; channel < c2_output_channel; ++channel){
    //         printf("channel %d\n",channel);
    //         for (uint8_t i = 0; i < c2_maxpool_H_out; ++i) {
    //             // printf("i %d\n",i);
    //             for (uint8_t j = 0; j < c2_maxpool_W_out; ++j) {
    //                 printf("%d,", c2_maxpool_out[((batch * batch_amt + channel) * c2_maxpool_H_out + i) * c2_maxpool_W_out + j]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }

    // C3
    const int c3_H_out = (c2_maxpool_H_out + 2*c3_padding - c_kernel_size)/c_stride + 1;
    const int c3_W_out = (c2_maxpool_W_out + 2*c3_padding - c_kernel_size)/c_stride + 1;
    uint8_t c3_output[batch_amt* c3_output_channel* c3_H_out* c3_W_out];
    
    conv2d(
        (uint8_t *)c2_maxpool_out,
        batch_amt, c2_output_channel, c2_maxpool_H_out, c2_maxpool_W_out,
        (int8_t *)c3_weight,
        c3_output_channel, c_kernel_size,
        (uint8_t *)c3_output,
        (int32_t *)c3_bias,
        c3_real_scale,
        c2_zero_point, 0, c3_zero_point,
        c_stride, c3_padding
    );
    relu_quant(
        (uint8_t*) c3_output,
        batch_amt, c3_output_channel, c3_H_out, c3_H_out,
        c3_zero_point
    );
    // printf("c3 result:\n");
    // for (uint8_t batch = 0; batch < batch_amt; ++batch){
    //     printf("batch %d\n", batch);
    //     for (uint8_t channel = 0; channel < c3_output_channel; ++channel){
    //         // printf("channel %d\n",channel);
    //         for (uint8_t i = 0; i < c3_H_out; ++i) {
    //             // printf("i %d\n",i);
    //             for (uint8_t j = 0; j < c3_W_out; ++j) {
    //                 printf("%d, ", c3_output[((batch * batch_amt + channel) * c3_H_out + i) * c3_W_out + j]);
    //             }
    //         }
    //     }
    // }
    uint8_t fc1_output[fc1_feature_out]; 
    fully_connected(
        (uint8_t *) c3_output,
        batch_amt, c3_output_channel,
        (uint8_t *)fc1_weight,
        fc1_feature_out,
        (int32_t *)fc1_bias,
        (uint8_t *)fc1_output,
        fc1_real_scale,
        fc1_zero_point, c3_zero_point, 0
    );
    relu_quant(
        (uint8_t*) fc1_output,
        batch_amt, fc1_feature_out, 1, 1,
        fc1_zero_point
    );
    // printf("fc1 result:\n");
    // for (uint8_t batch = 0; batch < batch_amt; ++batch){
    //     printf("batch %d\n", batch);
    //     for (uint8_t channel = 0; channel < fc1_feature_out; ++channel){
    //                 printf("%d, ", fc1_output[((batch * batch_amt + channel))]);
    //     }
    // }
    uint8_t fc2_output[fc2_feature_out]; 
    fully_connected(
        (uint8_t *) fc1_output,
        batch_amt, fc1_feature_out,
        (uint8_t *)fc2_weight,
        fc2_feature_out,
        (int32_t *)fc2_bias,
        (uint8_t *)fc2_output,
        fc2_real_scale,
        fc2_zero_point, fc1_zero_point, 0
    );
    printf("fc2 result:\n");
    for (uint8_t batch = 0; batch < batch_amt; ++batch){
        printf("batch %d\n", batch);
        for (uint8_t channel = 0; channel < fc2_feature_out; ++channel){
                    printf("%d, ", fc2_output[((batch * batch_amt + channel))]);
        }
    }
    return 0;
}