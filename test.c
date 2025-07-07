#include <stdio.h>
#include <stdint.h>
#include "test_images.c"
#include "calculations.c"
#include "weights.c"
#define batch_amt 1
#define img_input_channel 1
#define img_H 28
#define img_W 28
#define c1_output_channel 6
#define c1_kernel_size 5
#define c1_stride 1
#define c1_padding 2
#define c1_H_in 28
#define c1_W_in 28
#define c1_maxpool_stride 2
#define c1_maxpool_K 2

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
    // for (uint8_t batch = 0; batch < img_input_batch; ++batch){
    //     for (uint8_t channel = 0; channel < img_input_channel; ++channel){
    //         for (uint8_t i = 0; i < img_H; ++i) {
    //             for (uint8_t j = 0; j < img_W; ++j) {
    //                 printf("%d,", quantized_img[((batch * img_input_channel + channel) * img_H + i) * img_W + j]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }


    const int c1_H_out = (img_H + 2*c1_padding - c1_kernel_size)/c1_stride + 1;
    const int c1_W_out = (img_W + 2*c1_padding - c1_kernel_size)/c1_stride + 1;
    uint8_t c1_output[batch_amt* c1_output_channel *c1_H_out * c1_W_out];
    conv2d(
        (uint8_t *)quantized_img,
        batch_amt, img_input_channel, img_H, img_W,
        (int8_t *)c1_weight, c1_output_channel, c1_kernel_size,
        // &(c1_weight[0][0][0][0]), c1_output_channel, 5, 
        (uint8_t *)c1_output,
        // &(output[0][0][0][0]),
        (int32_t *)c1_bias,
        // &(c1_bias[0]),
        &c1_multiplier, &c1_right_shift, 
        &c1_zero_point, quant_zero_point, 0,
        c1_stride, c1_padding
    );

    // UNCOMMENT TO VIEW C1 RESULT
    // printf("c1 result:\n");
    // for (uint8_t batch = 0; batch < c1_batch_amt; ++batch){
    //     printf("batch %d\n", batch);
    //     for (uint8_t channel = 0; channel < c1_output_channel; ++channel){
    //         printf("channel %d\n",channel);
    //         for (uint8_t i = 0; i < c1_H_out; ++i) {
    //             // printf("i %d\n",i);
    //             for (uint8_t j = 0; j < c1_W_out; ++j) {
    //                 printf("%d,", c1_output[((batch * c1_batch_amt + channel) * c1_H_out + i) * c1_W_out + j]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }
    
    uint8_t c1_maxpool_H_out = (c1_H_out - 2) / c1_maxpool_stride + 1;
    uint8_t c1_maxpool_W_out = (c1_W_out - 2) / c1_maxpool_stride + 1;
    uint8_t c1_maxpool_out[batch_amt* c1_output_channel* c1_maxpool_H_out* c1_maxpool_W_out];
    maxpool2d(
            (uint8_t *)c1_output,
            c1_output_channel, c1_H_out, c1_W_out,
            c1_maxpool_K, c1_maxpool_stride,
            (uint8_t *)c1_maxpool_out
        );
    // UNCOMMENT TO VIEW C1 MAXPOOL RESULT
    printf("c1_maxpool result:\n");
    for (uint8_t batch = 0; batch < batch_amt; ++batch){
        printf("batch %d\n", batch);
        for (uint8_t channel = 0; channel < c1_output_channel; ++channel){
            printf("channel %d\n",channel);
            for (uint8_t i = 0; i < c1_maxpool_H_out; ++i) {
                // printf("i %d\n",i);
                for (uint8_t j = 0; j < c1_maxpool_W_out; ++j) {
                    printf("%d,", c1_maxpool_out[((batch * batch_amt + channel) * c1_maxpool_H_out + i) * c1_maxpool_W_out + j]);
                }
                printf("\n");
            }
        }
    }
    return 0;
}