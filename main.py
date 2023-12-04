from dct import dct_block, idct_block
from calculate import calculate_bpp, PSNR, MSE, Compression_Ratio, SSIM
from huffman import decode, encode, makenodes, iterate
from zigzag import zig_zag, zig_zag_reverse
import cv2
import collections
import os
import numpy as np
import time
from math import log10
import pandas as pd
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim


directory = './data/'

image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# image_files = ['vidu_1.png']

sorted_image_files = []
mse_values = []
psnr_values = []
ssim_values = []
compression_ratio_values = []
bpp_values = []
sorted_psnr_values = []
sorted_bpp_values = []
sorted_ssim_values = []
sorted_comratio_values = []
image_names = []
time_compress = []
sorted_time_compress = []

for image_file in image_files:
    # Construct the full file path
    filepath = os.path.join(directory, image_file)

    # Read the image
    image = cv2.imread(filepath)
    # image = cv2.imread(filepath)

    # Pad the image to make it divisible by 8 with white padding
    oHeight, oWidth = image.shape[:2]
    pad_height = (8 - oHeight % 8) % 8
    pad_width = (8 - oWidth % 8) % 8
    image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=255)

    # BGR to YCrBr
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Tách các kênh màu
    y_channel, cr_channel, cb_channel = cv2.split(ycbcr_image)

    img = y_channel

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

    iHeight, iWidth = image.shape[:2]
    zigZag = []

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = []
    #     for startY in range(0, iHeight, 8):
    #         for startX in range(0, iWidth, 8):
    #             block = img[startY:startY + 8, startX:startX + 8]

    #             # Tính DCT cho khối
    #             block_t = np.float32(block)  
    #             dct = dct_block(block_t)

    #             # lượng tử hóa các hệ số DCT
    #             block_q = np.floor(np.divide(dct, qtable) + 0.5)

    #             futures.append(executor.submit(zig_zag, block_q, 8))

    #     concurrent.futures.wait(futures)

    #     # Retrieve results
    #     for future in futures:
    #         zigZag.append(future.result())

    for startY in range(0, iHeight, 8):
        for startX in range(0, iWidth, 8):
            block = img[startY:startY + 8, startX:startX + 8]

            # Tính DCT cho khối
            block_t = np.float32(block)  # chuyển đổi sang số thực
            dct = dct_block(block_t)

            # lượng tử hóa các hệ số DCT
            block_q = np.floor(np.divide(dct, qtable) + 0.5)

            # Zig Zag
            zigZag.append(zig_zag(block_q, 8))
            
    # DPCM cho giá trị DC
    dc = []
    dc.append(zigZag[0][0])  # giữ nguyên giá trị đầu tiên
    for i in range(1, len(zigZag)):
        dc.append(zigZag[i][0] - zigZag[i - 1][0])

    # RLC cho giá trị AC
    rlc = []
    zeros = 0
    for i in range(0, len(zigZag)):
        zeros = 0
        for j in range(1, len(zigZag[i])):
            if (zigZag[i][j] == 0):
                zeros += 1
            else:
                rlc.append(zeros)
                rlc.append(zigZag[i][j])
                zeros = 0
        if (zeros != 0):
            rlc.append(zeros)
            rlc.append(0)
    #### Huffman ####

    # Huffman DPCM
    # Tìm tần suất xuất hiện cho mỗi giá trị của danh sách
    counterDPCM = collections.Counter(dc)

    # Xác định danh sách các giá trị dưới dạng danh sách các cặp (điểm, Tần suất tương ứng)
    probsDPCM = []
    for key, value in counterDPCM.items():
        probsDPCM.append((key, np.float32(value)))

    # Tạo danh sách các nút cho thuật toán Huffman
    symbolsDPCM = makenodes(probsDPCM)

    # chạy thuật toán Huffman trên một danh sách các "nút". Nó trả về một con trỏ đến gốc của một cây mới của "các nút bên trong".
    rootDPCM = iterate(symbolsDPCM)

    # Mã hóa danh sách các ký hiệu nguồn.
    sDPMC = encode(dc, symbolsDPCM)

    # Huffman RLC
    # Tìm tần suất xuất hiện cho mỗi giá trị của danh sách
    counterRLC = collections.Counter(rlc)

    # Xác định danh sách giá trị dưới dạng danh sách các cặp (điểm, Tần suất tương ứng)
    probsRLC = []
    for key, value in counterRLC.items():
        probsRLC.append((key, np.float32(value)))

    # Tạo danh sách các nút cho thuật toán Huffman
    symbolsRLC = makenodes(probsRLC)

    # chạy thuật toán Huffman trên một danh sách các "nút". Nó trả về một con trỏ đến gốc của một cây mới của "các nút bên trong".
    root = iterate(symbolsRLC)

    # Mã hóa danh sách các ký hiệu nguồn.
    sRLC = encode(rlc, symbolsRLC)
    stop = time.time()  # thời gian kết thúc nén

    ################## JPEG decompression ##################

    #### Huffman ####

    # Huffman DPCM
    # Giải mã một chuỗi nhị phân bằng cách sử dụng cây Huffman được truy cập thông qua root
    dDPCM = decode(sDPMC, rootDPCM)
    decodeDPMC = []
    for i in range(0, len(dDPCM)):
        decodeDPMC.append(float(dDPCM[i]))

    # Huffman RLC
    # Giải mã một chuỗi nhị phân bằng cách sử dụng cây Huffman được truy cập thông qua root
    dRLC = decode(sRLC, root)
    decodeRLC = []
    for i in range(0, len(dRLC)):
        decodeRLC.append(float(dRLC[i]))

    # Inverse DPCM
    inverse_DPCM = []
    inverse_DPCM.append(decodeDPMC[0])  # giá trị đầu tiên giữ nguyên
    for i in range(1, len(decodeDPMC)):
        inverse_DPCM.append(decodeDPMC[i] + inverse_DPCM[i - 1])

    # Inverse RLC
    inverse_RLC = []
    for i in range(0, len(decodeRLC)):
        if (i % 2 == 0):
            if (decodeRLC[i] != 0.0):
                if (i + 1 < len(decodeRLC) and decodeRLC[i + 1] == 0):
                    for j in range(1, int(decodeRLC[i])):
                        inverse_RLC.append(0.0)
                else:
                    for j in range(0, int(decodeRLC[i])):
                        inverse_RLC.append(0.0)
        else:
            inverse_RLC.append(decodeRLC[i])
    new_img = np.empty(shape=(iHeight, iWidth))
    height = 0
    width = 0
    temp = []
    temp2 = []
    for i in range(len(inverse_DPCM)):
        temp.append(inverse_DPCM[i])
        for j in range(0, 63):
            temp.append((inverse_RLC[j + i * 63]))
        temp2.append(temp)

        # inverse Zig-Zag và nghịch đảo Lượng tử hóa các hệ số DCT
        inverse_blockq = np.multiply(np.reshape(
            zig_zag_reverse(temp2), (8, 8)), qtable)

        # inverse DCT
        inverse_dct = idct_block(inverse_blockq)

        # Update new_img
        new_img[height:height + 8, width:width + 8] = inverse_dct

        # Update indices
        width += 8
        if width >= iWidth:
            width = 0
            height += 8

        temp = []
        temp2 = []

    np.place(new_img, new_img > 255, 255)
    np.place(new_img, new_img < 0, 0)

    ################ Hiển thị ảnh ##################
    # Gộp 3 kênh màu Y, Cr, Cb
    reconstructed_image_ycbcr = cv2.merge([img, cr_channel, cb_channel])

    # Chuyển đổi ảnh YCbCr lại thành ảnh màu
    new_img = cv2.cvtColor(reconstructed_image_ycbcr, cv2.COLOR_YCrCb2RGB)

    # Hiển thị ảnh gốc và ảnh giải nén
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(new_img), plt.axis('off'), plt.title('Reconstructed Image')
    # plt.subplot(123), plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Recovered Image')
    plt.show()

    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    # Lưu ảnh sau khi giải nén
    cv2.imwrite('decompressed.jpg', new_img)
    
    mse = MSE(image, new_img)
    psnr = PSNR(mse)
    ssim = SSIM(image, new_img)
    compression_ratio = Compression_Ratio(filepath)
    bpp = calculate_bpp(image, compression_ratio)
    image_files = [os.path.basename(filepath) for filepath in image_files]
    
    mse_values.append(mse)
    psnr_values.append(psnr)
    ssim_values.append(ssim)
    time_compress.append(stop-start)
    compression_ratio_values.append(compression_ratio)
    bpp_values.append(bpp)
    
    sorted_indices = np.argsort(bpp_values)
    sorted_bpp_values = np.array(bpp_values)[sorted_indices]
    sorted_psnr_values = np.array(psnr_values)[sorted_indices]
    sorted_ssim_values = np.array(ssim_values)[sorted_indices]
    sorted_comratio_values = np.array(compression_ratio_values)[sorted_indices]
    sorted_image_files = np.array(image_files)[sorted_indices]
    sorted_time_compress = np.array(time_compress)[sorted_indices]

    print(filepath)
    print('Time Compress: ',stop - start)
    print('Compression Ratio: ', compression_ratio)
    print("loading...")

data = {
'Image': sorted_image_files,
'BPP':  sorted_bpp_values,
'PSNR': sorted_psnr_values,
'SSIM': sorted_ssim_values,
'Compression Ratio': sorted_comratio_values,
'Time Compress' : sorted_time_compress,
}

pd.set_option('display.max_columns', None)
df = pd.DataFrame(data)
print(df)

sum = np.sum(sorted_time_compress)
print("Sum of time", sum)

plt.figure(figsize=(8, 6))
plt.plot(sorted_bpp_values, sorted_psnr_values, linestyle='-', marker='o', color='green', label='Line of Best Fit')
plt.title('PSNR vs Bits per Pixel (BPP) for Image Reconstruction')
plt.xlabel('Bits per Pixel (BPP)')
plt.ylabel('PSNR')
plt.grid(True)
plt.legend()
plt.show()
    
