import os
import cv2
import collections
import numpy as np
import time
from math import log
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

import huffman
import our_dct
import zigzag

def MSE(img1, img2):
    return ((img1.astype(np.float64) - img2.astype(np.float64)) ** 2).mean(axis=None)


def PSNR(mse):
    return 10 * log(((255 * 255) / mse), 10)

def SSIM(img1, img2):
    # Chuyển đổi sang kiểu float64 để tính toán chính xác
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Tính toán SSIM cho từng kênh màu và trung bình kết quả
    ssim_values = []
    for channel in range(img1.shape[-1]):
        channel_ssim = ssim(img1[:, :, channel], img2[:, :, channel], data_range=img2.max() - img2.min())
        ssim_values.append(channel_ssim)

    # Trả về giá trị SSIM trung bình của tất cả các kênh
    return np.mean(ssim_values)


def Compression_Ratio(filepath):
    Ori_img = os.stat(filepath).st_size
    Ori_img = Ori_img / 1024
    Com_img = os.path.getsize('decompressed.jpg')
    Com_img = Com_img / 1024
    CR = Ori_img / float(Com_img)
    return CR


def main():
    filepath = ('vidu.jpg')
    image = cv2.imread(filepath)

    # BGR to YCrBr
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Tách các kênh màu
    y_channel, cr_channel, cb_channel = cv2.split(ycbcr_image)

    #img = y_channel

    # Ma trận lượng tử hóa
    qtable = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                       [12, 12, 14, 19, 26, 58, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56],
                       [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77],
                       [24, 35, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101],
                       [72, 92, 95, 98, 112, 100, 103, 99]])

    qchrom = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                       [18, 21, 26, 66, 99, 99, 99, 99],
                       [24, 26, 56, 99, 99, 99, 99, 99],
                       [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])

    ################## JPEG compression ##################
    start = time.time()
    #iHeight, iWidth = img.shape[:2]
    iHeight, iWidth = image.shape[:2]
    zigZag_y = []
    zigZag_cb = []
    zigZag_cr = []
    for startY in range(0, iHeight, 8):
        for startX in range(0, iWidth, 8):
            # Lấy block từng khối
            block_y = y_channel[startY:startY + 8, startX:startX + 8]
            block_cb = cb_channel[startY:startY + 8, startX:startX + 8]
            block_cr = cr_channel[startY:startY + 8, startX:startX + 8]

            # Tính DCT cho khối
            block_t_y = np.float32(block_y)
            block_t_cb = np.float32(block_cb)
            block_t_cr = np.float32(block_cr)

            dct_y = our_dct.dct_block(block_t_y)
            dct_cb = our_dct.dct_block(block_t_cb)
            dct_cr = our_dct.dct_block(block_t_cr)

            # lượng tử hóa các hệ số DCT
            block_q_y = np.floor(np.divide(dct_y, qtable) + 0.5)
            block_q_cb = np.floor(np.divide(dct_cb, qchrom) + 0.5)
            block_q_cr = np.floor(np.divide(dct_cr, qchrom) + 0.5)

            # Zig Zag
            zigZag_y.append(zigzag.zig_zag(block_q_y, 8))
            zigZag_cb.append(zigzag.zig_zag(block_q_cb, 8))
            zigZag_cr.append(zigzag.zig_zag(block_q_cr, 8))

    # DPCM cho giá trị DC
    dc_y = []
    dc_y.append(zigZag_y[0][0])  # giữ nguyên giá trị đầu tiên
    for i in range(1, len(zigZag_y)):
        dc_y.append(zigZag_y[i][0] - zigZag_y[i - 1][0])
    dc_cb = []
    dc_cb.append(zigZag_cb[0][0])  # giữ nguyên giá trị đầu tiên
    for i in range(1, len(zigZag_cb)):
        dc_cb.append(zigZag_cb[i][0] - zigZag_cb[i - 1][0])
    dc_cr = []
    dc_cr.append(zigZag_cr[0][0])  # giữ nguyên giá trị đầu tiên
    for i in range(1, len(zigZag_cr)):
        dc_cr.append(zigZag_cr[i][0] - zigZag_cr[i - 1][0])

    # RLC cho giá trị AC
    rlc_y = []
    zeros = 0
    for i in range(0, len(zigZag_y)):
        zeros = 0
        for j in range(1, len(zigZag_y[i])):
            if (zigZag_y[i][j] == 0):
                zeros += 1
            else:
                rlc_y.append(zeros)
                rlc_y.append(zigZag_y[i][j])
                zeros = 0
        if (zeros != 0):
            rlc_y.append(zeros)
            rlc_y.append(0)
    rlc_cb = []
    zeros = 0
    for i in range(0, len(zigZag_cb)):
        zeros = 0
        for j in range(1, len(zigZag_cb[i])):
            if (zigZag_cb[i][j] == 0):
                zeros += 1
            else:
                rlc_cb.append(zeros)
                rlc_cb.append(zigZag_cb[i][j])
                zeros = 0
        if (zeros != 0):
            rlc_cb.append(zeros)
            rlc_cb.append(0)
    rlc_cr = []
    zeros = 0
    for i in range(0, len(zigZag_cr)):
        zeros = 0
        for j in range(1, len(zigZag_cr[i])):
            if (zigZag_cr[i][j] == 0):
                zeros += 1
            else:
                rlc_cr.append(zeros)
                rlc_cr.append(zigZag_cr[i][j])
                zeros = 0
        if (zeros != 0):
            rlc_cr.append(zeros)
            rlc_cr.append(0)
    #### Huffman ####

    # Huffman DPCM
    # Tìm tần suất xuất hiện cho mỗi giá trị của danh sách
    counterDPCM_y = collections.Counter(dc_y)
    counterDPCM_cb = collections.Counter(dc_cb)
    counterDPCM_cr = collections.Counter(dc_cr)

    # Xác định danh sách các giá trị dưới dạng danh sách các cặp (điểm, Tần suất tương ứng)
    probsDPCM_y = []
    for key, value in counterDPCM_y.items():
        probsDPCM_y.append((key, np.float32(value)))
    probsDPCM_cb = []
    for key, value in counterDPCM_cb.items():
        probsDPCM_cb.append((key, np.float32(value)))
    probsDPCM_cr = []
    for key, value in counterDPCM_cr.items():
        probsDPCM_cr.append((key, np.float32(value)))

    # Tạo danh sách các nút cho thuật toán Huffman
    symbolsDPCM_y = huffman.makenodes(probsDPCM_y)
    symbolsDPCM_cb = huffman.makenodes(probsDPCM_cb)
    symbolsDPCM_cr = huffman.makenodes(probsDPCM_cr)

    # chạy thuật toán Huffman trên một danh sách các "nút". Nó trả về một con trỏ đến gốc của một cây mới của "các nút bên trong".
    rootDPCM_y = huffman.iterate(symbolsDPCM_y)
    rootDPCM_cb = huffman.iterate(symbolsDPCM_cb)
    rootDPCM_cr = huffman.iterate(symbolsDPCM_cr)

    # Mã hóa danh sách các ký hiệu nguồn.
    sDPMC_y = huffman.encode(dc_y, symbolsDPCM_y)
    sDPMC_cb = huffman.encode(dc_cb, symbolsDPCM_cb)
    sDPMC_cr = huffman.encode(dc_cr, symbolsDPCM_cr)

    # Huffman RLC
    # Tìm tần suất xuất hiện cho mỗi giá trị của danh sách
    counterRLC_y = collections.Counter(rlc_y)
    counterRLC_cb = collections.Counter(rlc_cb)
    counterRLC_cr = collections.Counter(rlc_cr)

    # Xác định danh sách giá trị dưới dạng danh sách các cặp (điểm, Tần suất tương ứng)
    probsRLC_y = []
    for key, value in counterRLC_y.items():
        probsRLC_y.append((key, np.float32(value)))
    probsRLC_cb = []
    for key, value in counterRLC_cb.items():
        probsRLC_cb.append((key, np.float32(value)))
    probsRLC_cr = []
    for key, value in counterRLC_cr.items():
        probsRLC_cr.append((key, np.float32(value)))

    # Tạo danh sách các nút cho thuật toán Huffman
    symbolsRLC_y = huffman.makenodes(probsRLC_y)
    symbolsRLC_cb = huffman.makenodes(probsRLC_cb)
    symbolsRLC_cr = huffman.makenodes(probsRLC_cr)

    # chạy thuật toán Huffman trên một danh sách các "nút". Nó trả về một con trỏ đến gốc của một cây mới của "các nút bên trong".
    root_y = huffman.iterate(symbolsRLC_y)
    root_cb = huffman.iterate(symbolsRLC_cb)
    root_cr = huffman.iterate(symbolsRLC_cr)

    # Mã hóa danh sách các ký hiệu nguồn.
    sRLC_y = huffman.encode(rlc_y, symbolsRLC_y)
    sRLC_cb = huffman.encode(rlc_cb, symbolsRLC_cb)
    sRLC_cr = huffman.encode(rlc_cr, symbolsRLC_cr)
    stop = time.time()  # thời gian kết thúc nén

    ################## JPEG decompression ##################

    #### Huffman ####

    # Huffman DPCM
    # Giải mã một chuỗi nhị phân bằng cách sử dụng cây Huffman được truy cập thông qua root
    dDPCM_y = huffman.decode(sDPMC_y, rootDPCM_y)
    dDPCM_cb = huffman.decode(sDPMC_cb, rootDPCM_cb)
    dDPCM_cr = huffman.decode(sDPMC_cr, rootDPCM_cr)

    decodeDPMC_y = []
    for i in range(0, len(dDPCM_y)):
        decodeDPMC_y.append(float(dDPCM_y[i]))
    decodeDPMC_cb = []
    for i in range(0, len(dDPCM_cb)):
        decodeDPMC_cb.append(float(dDPCM_cb[i]))
    decodeDPMC_cr = []
    for i in range(0, len(dDPCM_cr)):
        decodeDPMC_cr.append(float(dDPCM_cr[i]))

    # Huffman RLC
    # Giải mã một chuỗi nhị phân bằng cách sử dụng cây Huffman được truy cập thông qua root
    dRLC_y = huffman.decode(sRLC_y, root_y)
    decodeRLC_y = []
    for i in range(0, len(dRLC_y)):
        decodeRLC_y.append(float(dRLC_y[i]))

    dRLC_cb = huffman.decode(sRLC_cb, root_cb)
    decodeRLC_cb = []
    for i in range(0, len(dRLC_cb)):
        decodeRLC_cb.append(float(dRLC_cb[i]))

    dRLC_cr = huffman.decode(sRLC_cr, root_cr)
    decodeRLC_cr = []
    for i in range(0, len(dRLC_cr)):
        decodeRLC_cr.append(float(dRLC_cr[i]))

    # Inverse DPCM
    inverse_DPCM_y = []
    inverse_DPCM_y.append(decodeDPMC_y[0])  # giá trị đầu tiên giữ nguyên
    for i in range(1, len(decodeDPMC_y)):
        inverse_DPCM_y.append(decodeDPMC_y[i] + inverse_DPCM_y[i - 1])

    inverse_DPCM_cb = []
    inverse_DPCM_cb.append(decodeDPMC_cb[0])  # giá trị đầu tiên giữ nguyên
    for i in range(1, len(decodeDPMC_cb)):
        inverse_DPCM_cb.append(decodeDPMC_cb[i] + inverse_DPCM_cb[i - 1])

    inverse_DPCM_cr = []
    inverse_DPCM_cr.append(decodeDPMC_cr[0])  # giá trị đầu tiên giữ nguyên
    for i in range(1, len(decodeDPMC_cr)):
        inverse_DPCM_cr.append(decodeDPMC_cr[i] + inverse_DPCM_cr[i - 1])

    # Inverse RLC
    inverse_RLC_y = []
    for i in range(0, len(decodeRLC_y)):
        if (i % 2 == 0):
            if (decodeRLC_y[i] != 0.0):
                if (i + 1 < len(decodeRLC_y) and decodeRLC_y[i + 1] == 0):
                    for j in range(1, int(decodeRLC_y[i])):
                        inverse_RLC_y.append(0.0)
                else:
                    for j in range(0, int(decodeRLC_y[i])):
                        inverse_RLC_y.append(0.0)
        else:
            inverse_RLC_y.append(decodeRLC_y[i])

    inverse_RLC_cb = []
    for i in range(0, len(decodeRLC_cb)):
        if (i % 2 == 0):
            if (decodeRLC_cb[i] != 0.0):
                if (i + 1 < len(decodeRLC_cb) and decodeRLC_cb[i + 1] == 0):
                    for j in range(1, int(decodeRLC_cb[i])):
                        inverse_RLC_cb.append(0.0)
                else:
                    for j in range(0, int(decodeRLC_cb[i])):
                        inverse_RLC_cb.append(0.0)
        else:
            inverse_RLC_cb.append(decodeRLC_cb[i])

    inverse_RLC_cr = []
    for i in range(0, len(decodeRLC_cr)):
        if (i % 2 == 0):
            if (decodeRLC_cr[i] != 0.0):
                if (i + 1 < len(decodeRLC_cr) and decodeRLC_cr[i + 1] == 0):
                    for j in range(1, int(decodeRLC_cr[i])):
                        inverse_RLC_cr.append(0.0)
                else:
                    for j in range(0, int(decodeRLC_cr[i])):
                        inverse_RLC_cr.append(0.0)
        else:
            inverse_RLC_cr.append(decodeRLC_cr[i])
    ##
    new_img_y = np.empty(shape=(iHeight, iWidth))
    height = 0
    width = 0
    temp = []
    temp2 = []
    for i in range(0, len(inverse_DPCM_y)):
        temp.append(inverse_DPCM_y[i])
        for j in range(0, 63):
            temp.append((inverse_RLC_y[j + i * 63]))
        temp2.append(temp)

        # inverse Zig-Zag và nghịch đảo Lượng tử hóa các hệ số DCT
        inverse_blockq = np.multiply(np.reshape(zigzag.zig_zag_reverse(temp2), (8, 8)), qtable)

        # inverse DCT
        inverse_dct = our_dct.idct_block(inverse_blockq)
        for startY in range(height, height + 8, 8):
            for startX in range(width, width + 8, 8):
                new_img_y[startY:startY + 8, startX:startX + 8] = inverse_dct
        width = width + 8
        if (width == iHeight):
            width = 0
            height = height + 8
        temp = []
        temp2 = []

    np.place(new_img_y, new_img_y > 255, 255)
    np.place(new_img_y, new_img_y < 0, 0)
    ##
    new_img_cb = np.empty(shape=(iHeight, iWidth))
    height = 0
    width = 0
    temp = []
    temp2 = []
    for i in range(0, len(inverse_DPCM_cb)):
        temp.append(inverse_DPCM_cb[i])
        for j in range(0, 63):
            temp.append((inverse_RLC_cb[j + i * 63]))
        temp2.append(temp)

        # inverse Zig-Zag và nghịch đảo Lượng tử hóa các hệ số DCT
        inverse_blockq = np.multiply(np.reshape(
            zigzag.zig_zag_reverse(temp2), (8, 8)), qchrom)

        # inverse DCT
        inverse_dct = our_dct.idct_block(inverse_blockq)
        for startY in range(height, height + 8, 8):
            for startX in range(width, width + 8, 8):
                new_img_cb[startY:startY + 8, startX:startX + 8] = inverse_dct
        width = width + 8
        if (width == iHeight):
            width = 0
            height = height + 8
        temp = []
        temp2 = []

    np.place(new_img_cb, new_img_cb > 255, 255)
    np.place(new_img_cb, new_img_cb < 0, 0)

    ##
    new_img_cr = np.empty(shape=(iHeight, iWidth))
    height = 0
    width = 0
    temp = []
    temp2 = []
    for i in range(0, len(inverse_DPCM_cr)):
        temp.append(inverse_DPCM_cr[i])
        for j in range(0, 63):
            temp.append((inverse_RLC_cr[j + i * 63]))
        temp2.append(temp)

        # inverse Zig-Zag và nghịch đảo Lượng tử hóa các hệ số DCT
        inverse_blockq = np.multiply(np.reshape(
            zigzag.zig_zag_reverse(temp2), (8, 8)), qchrom)

        # inverse DCT
        inverse_dct = our_dct.idct_block(inverse_blockq)
        for startY in range(height, height + 8, 8):
            for startX in range(width, width + 8, 8):
                new_img_cr[startY:startY + 8, startX:startX + 8] = inverse_dct
        width = width + 8
        if (width == iHeight):
            width = 0
            height = height + 8
        temp = []
        temp2 = []

    np.place(new_img_cr, new_img_cr > 255, 255)
    np.place(new_img_cr, new_img_cr < 0, 0)

    ################ Hiển thị ảnh ##################
    # Gộp 3 kênh màu Y, Cr, Cb
    #reconstructed_image_ycbcr = cv2.merge([img, cr_channel, cb_channel])
    reconstructed_image_ycbcr = cv2.merge([new_img_y, new_img_cr, new_img_cb])

    # Chuyển đổi ảnh YCbCr lại thành ảnh RGB
    reconstructed_image_ycbcr_uint8 = reconstructed_image_ycbcr.astype(np.uint8)

    new_img = cv2.cvtColor(reconstructed_image_ycbcr_uint8, cv2.COLOR_YCrCb2RGB)

    # Hiển thị ảnh gốc và ảnh giải nén
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(new_img), plt.axis('off'), plt.title('Reconstructed Image')
    plt.show()

    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    # Lưu ảnh sau khi giải nén
    cv2.imwrite('decompressed.jpg', new_img)

    # Tính MSE
    mse = MSE(image, new_img)
    print("MSE = ", mse)

    # Tính PSNR
    print("PSNR = ", PSNR(mse))

    # Tính SSIM
    print("SSIM = ", SSIM(image, new_img))

    # Compression Ratio
    print("Compression Ratio = ", Compression_Ratio(filepath))

    # Thời gian nén
    print("Time Compress: ", stop - start)


if __name__ == "__main__":
    main()
