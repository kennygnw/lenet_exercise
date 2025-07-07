#include <stdint.h>

static inline uint8_t round_to_uint8(float x)
{
    return (uint8_t)(x + (x >= 0 ? 0.5f : -0.5f));
}
/* --- small helper: single-value requantiser ----------------------------- */
static inline uint8_t requantize_fixed_point(
        int32_t src,
        int32_t multiplier,
        int8_t right_shift,
        uint8_t zero_point
    )
{
    int64_t prod     = (int64_t)src * (int64_t)multiplier;
    int64_t nudge    = (int64_t)1 << (right_shift > 0 ? (right_shift - 1) : 0);
    int64_t rounded  = prod + nudge;
    // int32_t requant  = (int32_t)(rounded >> right_shift);
    int32_t requant  = zero_point + (int32_t)(rounded >> right_shift);
    if (requant <   0) requant =   0;
    if (requant > 255) requant = 255;
    return (uint8_t)requant;
}

/* full convolution --------------------------------------------------- */
void conv2d(
    // padding value is 0
    /* input  [B, C_in, H_in, W_in]  */
    const uint8_t *input,
    int B, int C_in, int H_in, int W_in,    /* kernel tensor [C_out, C_in, K, K]  (flattened) */
    const int8_t  *kernel,
    int C_out, int K,
    /* output tensor [C_out, H_out, W_out] (flattened) */
    uint8_t       *output,
    /* per-output-channel parameters */
    const int32_t *bias,         /* [C_out]  */
    const int32_t *multiplier,   /* [C_out]  */
    const int8_t  *right_shift,  /* [C_out]  */
    const uint8_t *out_zp,       /* [C_out]  */
    /* zero-points common to every channel */
    uint8_t  x_zp,               /* input  zero-point */
    int8_t   w_zp,               /* weight zero-point (often 0) */
    /* convolution hyper-params */
    int stride,
    int padding)
{
    int H_out = (H_in + 2 * padding - K) / stride + 1;
    int W_out = (W_in + 2 * padding - K) / stride + 1;


    /* loop order:  b | oc | i | j | ic | ki | kj -------------------- */
    for (int b = 0; b < B; ++b)
    {
        const uint8_t *in_b  = input  + b * C_in  * H_in * W_in;
        uint8_t       *out_b = output + b * C_out * H_out * W_out;

        for (int oc = 0; oc < C_out; ++oc)
        {
            printf("channel %d\n", oc);
            int32_t  b_add   = bias[oc];
            // commented since currently the application is per tensor not per channel
            // int32_t  mul     = multiplier [oc];
            // int32_t  rshift  = (int32_t)right_shift[oc];
            // int32_t  zp_out  = (int32_t)out_zp[oc];

            for (int i = 0; i < H_out; ++i)
            {
                int start_i = i * stride - padding;
                for (int j = 0; j < W_out; ++j)
                {
                    int start_j = j * stride - padding;
                    int32_t sum = 0;

                    for (int ic = 0; ic < C_in; ++ic)
                    {
                        const uint8_t *in_c =
                            in_b + ic * H_in * W_in;

                        const int8_t  *w_base =
                            kernel + (((oc * C_in + ic) * K) * K);

                        for (int ki = 0; ki < K; ++ki)
                        {
                            int in_i = start_i + ki;
                            
                            if (in_i < 0 || in_i >= H_in) continue;

                            for (int kj = 0; kj < K; ++kj)
                            {
                                int in_j = start_j + kj;
                                if (in_j < 0 || in_j >= W_in) continue;

                                int32_t xv = (int32_t)
                                   in_c[in_i * W_in + in_j] - (int32_t)x_zp;

                                int32_t wv = (int32_t)
                                   w_base[ki * K + kj]       - (int32_t)w_zp;

                                sum += xv * wv;
                            }
                        }
                    }

                    sum += b_add;
                    out_b[(oc * H_out + i) * W_out + j] = requantize_fixed_point(sum, *multiplier, *right_shift, *out_zp);
                }
            }
        }
    }
    
    // /* nested loops:  oc | i | j | ic | ki | kj  ---------------------- */
    // for (int oc = 0; oc < C_out; ++oc)
    // {
    //     int32_t  b      = bias[oc];
    //     // commented since currently the application is per tensor not per channel
    //     // int32_t  mul    = multiplier  [oc];
    //     // int32_t  rshift = (int32_t)right_shift[oc];
    //     // int32_t  zp_out = (int32_t)out_zp[oc];

    //     for (int i = 0; i < H_out; ++i)
    //     {
    //         for (int j = 0; j < W_out; ++j)
    //         {
    //             int32_t sum = 0;

    //             int start_i = i * stride - padding;
    //             int start_j = j * stride - padding;

    //             for (int ic = 0; ic < C_in; ++ic)
    //             {
    //                 for (int ki = 0; ki < K; ++ki)
    //                 {
    //                     int in_i = start_i + ki;
    //                     if (in_i < 0 || in_i >= H_in) continue;

    //                     for (int kj = 0; kj < K; ++kj)
    //                     {
    //                         int in_j = start_j + kj;
    //                         if (in_j < 0 || in_j >= W_in) continue;

    //                         /* -------- fetch & zero-center ------------- */
    //                         int32_t xv = (int32_t)
    //                                      input[(ic*H_in + in_i)*W_in + in_j]
    //                                      - (int32_t)x_zp;
                            
    //                         int32_t wv = (int32_t)
    //                                      kernel[(((oc*C_in + ic)*K + ki)*K + kj)]
    //                                      - (int32_t)w_zp;

    //                         sum += xv * wv;
    //                     }
    //                 }
    //             }
    //             /* add bias and requantise -------------------------- */
    //             sum += b;
    //             output[(oc*H_out + i)*W_out + j] = requantize_fixed_point(sum, *multiplier, *right_shift, *out_zp);
    //         }
    //     }
    // }
}
void quantize_image(
    const unsigned char *input,
    uint8_t input_batch_amt, uint8_t img_channel,
    uint8_t img_H, uint8_t img_W,
    unsigned char *output,
    float quant_scale, uint8_t quant_zero_point
)
{   
    // size_t stride = (size_t)img_channel * img_H * img_W;   /* bytes per batch */
    float scale = quant_scale*255;
    for (uint8_t b = 0; b < input_batch_amt; ++b)
    {
        const uint8_t *in_batch  = input  + b * img_channel;
        // printf("%d ", input);
        
        uint8_t *out_batch = output + b * img_channel;
        for (uint8_t ch =0; ch < img_channel; ++ch) {
            const uint8_t *in_ch  = in_batch  + ch * img_H * img_W;
            uint8_t *out_ch = out_batch + ch * img_H * img_W;
            for (uint32_t idx = 0; idx < img_H * img_W; ++idx) {
                float quant_val = in_ch[idx] / scale;
                // printf("%d ",round_to_uint8(quant_val)+ quant_zero_point);
                out_ch[idx] = round_to_uint8(quant_val) + quant_zero_point;
            }
        }
    }
}

void maxpool2d(uint8_t *input,
                             int C_in, int H_in, int W_in,
                             int K, int stride,
                             uint8_t *output)
{
    int H_out = (H_in - K) / stride + 1;
    int W_out = (W_in - K) / stride + 1;
    // printf("H_out: %d, W_out: %d",H_out, W_out);
    for (int c = 0; c < C_in; ++c)
    {
        uint8_t  *in_base  = input  + c * H_in * W_in;
        uint8_t  *out_base = output + c * H_out * W_out;

        for (int i = 0; i < H_out; ++i)
        {
            int in_i0 = i * stride;

            for (int j = 0; j < W_out; ++j)
            {
                int in_j0 = j * stride;

                /* ---------- find max in K×K window ------------------- */
                uint8_t  max_val = in_base[(in_i0)*W_in + in_j0];

                for (int ki = 0; ki < K; ++ki)
                {
                    const uint8_t *row = in_base + (in_i0 + ki) * W_in + in_j0;
                    for (int kj = 0; kj < K; ++kj)
                    {
                        uint8_t v = row[kj];
                        if (v > max_val) max_val = v;
                    }
                }
                out_base[i * W_out + j] = max_val;
            }
        }
    }
}
void relu_quant(
    uint8_t* input,
    uint8_t input_batch_amt, uint8_t img_channel,
    uint8_t img_H, uint8_t img_W,
    uint8_t zero_point
)
{


}
