package com.example.androidapp

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.util.Base64
import android.util.Log
import java.io.ByteArrayOutputStream
import android.graphics.Bitmap

class ImageConverter {
    companion object {
        private const val TAG = "ImageConverter"

        /**
         * Converts a YUV_420_888 Image object to a high-quality JPEG byte array.
         * The Image is closed after conversion.
         */
        fun yuvToHighQualityJpeg(image: Image): ByteArray? {
            if (image.format != ImageFormat.YUV_420_888) {
                Log.e(TAG, "Unsupported image format: ${image.format}")
                image.close() // Important to close the image even on error
                return null
            }

            return try {
                // Get the planes from the image
                val planes = image.planes
                val yPlane = planes[0]
                val uPlane = planes[1]
                val vPlane = planes[2]

                // Get the buffers and their sizes
                val yBuffer = yPlane.buffer
                val uBuffer = uPlane.buffer
                val vBuffer = vPlane.buffer
                val ySize = yBuffer.remaining()
                val uSize = uBuffer.remaining()
                val vSize = vBuffer.remaining()

                // Create a single byte array to hold the YUV data in NV21 format
                val nv21 = ByteArray(ySize + uSize + vSize)

                // Copy Y-plane data
                yBuffer.get(nv21, 0, ySize)

                // Interleave U and V planes into the NV21 format
                // NV21 stores UV data as a single plane with V and U interleaved (VuVu)
                val uvIndex = ySize
                val uvWidth = image.width / 2
                val uvHeight = image.height / 2
                val uStride = uPlane.rowStride
                val vStride = vPlane.rowStride
                val uvPixelStride = uPlane.pixelStride

                for (j in 0 until uvHeight) {
                    for (i in 0 until uvWidth) {
                        val u = uBuffer.get(j * uStride + i * uvPixelStride).toInt() and 0xFF
                        val v = vBuffer.get(j * vStride + i * uvPixelStride).toInt() and 0xFF
                        nv21[uvIndex + j * uvWidth * 2 + i * 2] = v.toByte()
                        nv21[uvIndex + j * uvWidth * 2 + i * 2 + 1] = u.toByte()
                    }
                }

                // Create a YuvImage and compress it to JPEG
                val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
                val out = ByteArrayOutputStream()
                yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)

                return out.toByteArray()
            } catch (e: Exception) {
                Log.e(TAG, "Conversion error: ${e.message}", e)
                return null
            } finally {
                image.close() // Ensure the image is always closed
            }
        }

        /**
         * Splits a JPEG byte array into a list of Base64-encoded string fragments.
         */
        fun splitImageIntoFragments(jpegBytes: ByteArray, maxFragmentSize: Int): List<String> {
            val fragments = mutableListOf<String>()
            var offset = 0
            while (offset < jpegBytes.size) {
                val remainingBytes = jpegBytes.size - offset
                val fragmentSize = minOf(maxFragmentSize, remainingBytes)
                val chunk = jpegBytes.sliceArray(offset until offset + fragmentSize)
                val base64Chunk = Base64.encodeToString(chunk, Base64.NO_WRAP)
                fragments.add(base64Chunk)
                offset += fragmentSize
            }
            return fragments
        }

        /**
         * Converts a Bitmap to a JPEG byte array with a specified quality.
         */
        fun bitmapToJpegBytes(bitmap: Bitmap): ByteArray {
            val outputStream = ByteArrayOutputStream()
            // Use a consistent compression quality
            bitmap.compress(Bitmap.CompressFormat.JPEG, 70, outputStream)
            return outputStream.toByteArray()
        }
    }
}