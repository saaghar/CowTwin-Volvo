package com.example.androidapp

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.media.Image
import android.media.ImageReader
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.*
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.net.*
import java.nio.ByteBuffer
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.random.Random
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.util.UnstableApi
import androidx.media3.datasource.DataSource
import androidx.media3.datasource.UdpDataSource
import androidx.media3.exoplayer.DefaultLoadControl
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.LoadControl
import androidx.media3.exoplayer.source.ProgressiveMediaSource
import kotlin.math.*
import com.google.gson.Gson // Make sure to import Gson
import java.util.Stack

data class FrameData(
    val imageData: ByteArray
)

@Serializable
data class RequestForSignDetection(
    val TotalNumberOfFragments: Int,
    val UniqueSequencedID: String,
    val SignGpsPosition: String? = null
)

@Serializable
data class SendImageMessage(
    val UniqueSequencedID: String,
    val fragmentSequenceID: Int,
    val fragment: String,
    val SignGpsPosition: String? = null
)

@Serializable
data class ResponseForSignDetection(
    val UniqueSequencedID: String,
    val IsSignDetected: Boolean
)

@UnstableApi
class ImageUdpService : Service() {

    private val serviceJob = Job()
    private val serviceScope = CoroutineScope(Dispatchers.IO + serviceJob)

    private var exoPlayer: ExoPlayer? = null
    private var imageReader: ImageReader? = null
    private val FRAME_WIDTH = 640
    private val FRAME_HEIGHT = 480
    private val MAX_IMAGES_TO_BUFFER = 10
    private val UDP_STREAM_PORT = 5000

    private val frameStack = Stack<FrameData>()
    private val maxStackSize = 1

    private val frameQueue = ArrayBlockingQueue<FrameData>(200)
    private var isPlayerReady = AtomicBoolean(false)
    private var handlerThread: HandlerThread? = null

    private var tflite: Interpreter? = null
    private val MODEL_PATH = "best_binary_float16.tflite"
    private val IMAGE_SIZE = 640
    private val NORMALIZE_MEAN = 0.0f
    private val NORMALIZE_STD = 255.0f
    private val DETECTION_THRESHOLD = 0.5f

    private val CONFIDENCE_SCORE_INDEX_IN_CHUNKS = 4
    private val NUM_VALUES_PER_DETECTION = 6

    private val CAMERA_ALTITUDE_M = 2.8
    private val CAMERA_BEARING_DEG = 45.0
    private val CAMERA_TILT_DEG = -5.0

    private val FOCAL_LENGTH_MM = 3.43
    private val SENSOR_WIDTH_MM = 1920 * 3.0e-3
    private val SENSOR_HEIGHT_MM = 1088 * 3.0e-3

    private val SIGN_REAL_WIDTH_M = 0.6
    private val SIGN_REAL_HEIGHT_M = 0.6

    private var serviceLocationManager: LocationManager? = null
    private var serviceLocationListener: LocationListener? = null
    private var lastKnownLocation: AtomicReference<Location?> = AtomicReference(null)
    private var lastKnownBearing: AtomicReference<Float?> = AtomicReference(null)

    private val trackedSigns = mutableListOf<SignRecord>()

    data class SignRecord(
        val uniqueIdString: String,
        val signLat: Double,
        val signLon: Double,
        val distanceToSign: Double
    )

    companion object {
        const val NOTIFICATION_CHANNEL_ID = "ImageUdpServiceChannel"
        const val NOTIFICATION_ID = 1
        const val TAG = "ImageUdpService"
        const val SERVER_IP = "10.222.105.13"
        const val SERVER_PORT = 9876
        const val CHUNK_SIZE = 1496

        const val RESPONSE_TIMEOUT = 250L
        const val FRAME_PROCESSING_DELAY = 10L
        const val SIGN_DETECTED_PAUSE_MS = 500L
        const val LOCATION_UPDATE_INTERVAL_MS = 1000L
        const val LOCATION_MIN_DISTANCE_M = 1f
        const val SAME_SIGN_DISTANCE_EPSILON = 2

        const val NO_SIGN_DETECTED_DIRECTORY = "NoSignDetected"
    }

    private var udpSender: UdpSender? = null
    private val gson = Gson()
    private val json = Json { ignoreUnknownKeys = true }

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service onCreate. Initializing components.")

        try {
            val modelBuffer = FileUtil.loadMappedFile(this, MODEL_PATH)
            val options = Interpreter.Options()
            tflite = Interpreter(modelBuffer, options)
            Log.d(TAG, "TensorFlow Lite model loaded successfully: $MODEL_PATH")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading TensorFlow Lite model: $MODEL_PATH", e)
            stopSelf()
            return
        }

        handlerThread = HandlerThread("ImageReaderHandler")
        handlerThread?.start()
        val handler = Handler(handlerThread!!.looper)

        try {
            imageReader = ImageReader.newInstance(
                FRAME_WIDTH,
                FRAME_HEIGHT,
                ImageFormat.YUV_420_888,
                MAX_IMAGES_TO_BUFFER
            )
            var frameCounter = 0 // Initialize a counter
            imageReader?.setOnImageAvailableListener({ reader ->
                var image: Image? = null
                try {
                    image = reader.acquireNextImage()
                    if (image != null) {
                        frameCounter++
                        if (frameCounter % 4 == 0) {
                            val jpegBytes = ImageConverter.yuvToHighQualityJpeg(image)
                            if (jpegBytes != null) {
                                val frameData = FrameData(jpegBytes)

                                synchronized(frameStack) {
                                    // Remove oldest frames if stack is at capacity
                                    while (frameStack.size >= maxStackSize) {
                                        frameStack.removeAt(0) // Remove from bottom of stack
                                    }
                                    frameStack.push(frameData)
                                    Log.d(TAG, "Frame added to stack. Stack size: ${frameStack.size}")
                                }
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error in onImageAvailable: ${e.message}", e)
                } finally {
                    image?.close()
                }
            }, handler)

            Log.d(TAG, "ImageReader initialized for frame capture. Resolution: ${FRAME_WIDTH}x${FRAME_HEIGHT}")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing ImageReader: ${e.message}", e)
            stopSelf()
            return
        }

        serviceLocationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        startContinuousLocationUpdates()

        udpSender = UdpSender(
            serverIp = SERVER_IP,
            serverPort = SERVER_PORT,
            TAG = TAG,
            gson = gson
        )
        udpSender?.initialize()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForegroundService()
        Log.d(TAG, "Service started. Initiating ExoPlayer and frame processing.")

        serviceScope.launch {
            withContext(Dispatchers.Main) {
                startExoPlayer()
            }

            Log.i(TAG, "Main loop waiting for isPlayerReady to be true. Current state: ${isPlayerReady.get()}")
            while (!isPlayerReady.get() && serviceScope.isActive) {
                delay(100)
            }
            Log.i(TAG, "ExoPlayer is ready. Starting continuous frame processing.")

            processFramesContinuously()

            Log.d(TAG, "Service processing loop finished (should be continuous).")
            stopSelf()
        }

        return START_NOT_STICKY
    }

    @SuppressLint("MissingPermission")
    private fun startContinuousLocationUpdates() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            ContextCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            Log.e(TAG, "Service: Location permissions not granted. Cannot get continuous updates.")
            return
        }

        serviceLocationListener = object : LocationListener {
            override fun onLocationChanged(location: Location) {
                lastKnownLocation.set(location)
                if (location.hasBearing()) { // Check if the location has a valid bearing
                    lastKnownBearing.set(location.bearing)
                    Log.d(TAG, "Service: Location updated: Lat=${"%.4f".format(location.latitude)}, Lon=${"%.4f".format(location.longitude)}, Bearing=${"%.2f".format(location.bearing)}")
                } else {
                    Log.d(TAG, "Service: Location updated without a valid bearing.")
                }            }

            override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {
                Log.d(TAG, "Service: Location status changed for $provider: $status")
            }

            override fun onProviderEnabled(provider: String) {
                Log.d(TAG, "Service: Location provider enabled: $provider")
            }

            override fun onProviderDisabled(provider: String) {
                Log.w(TAG, "Service: Location provider disabled: $provider")
            }
        }

        serviceLocationManager?.let { manager ->
            val providers = mutableListOf<String>()
            if (manager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                providers.add(LocationManager.GPS_PROVIDER)
            }
            if (manager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                providers.add(LocationManager.NETWORK_PROVIDER)
            }

            if (providers.isEmpty()) {
                Log.e(TAG, "Service: No location providers enabled for continuous updates.")
                return
            }

            for (provider in providers) {
                manager.requestLocationUpdates(
                    provider,
                    LOCATION_UPDATE_INTERVAL_MS,
                    LOCATION_MIN_DISTANCE_M,
                    serviceLocationListener!!,
                    Looper.getMainLooper()
                )
                Log.d(TAG, "Service: Started continuous location updates from $provider.")
            }
        } ?: Log.e(TAG, "Service: LocationManager not initialized for continuous updates.")
    }

    private fun startExoPlayer() {
        val loadControl: LoadControl = DefaultLoadControl.Builder()
            .setBufferDurationsMs(5000, 15000, 2500, 5000)
            .setTargetBufferBytes(DefaultLoadControl.DEFAULT_TARGET_BUFFER_BYTES)
            .setPrioritizeTimeOverSizeThresholds(false)
            .build()

        exoPlayer = ExoPlayer.Builder(this)
            .setLoadControl(loadControl)
            .build()

        val imageReaderSurface = imageReader?.surface
        if (imageReaderSurface == null) {
            Log.e(TAG, "ImageReader surface is null. Cannot start ExoPlayer.")
            stopSelf()
            return
        }
        exoPlayer?.setVideoSurface(imageReaderSurface)

        val videoUri = Uri.parse("udp://0.0.0.0:$UDP_STREAM_PORT")
        val mediaItem = MediaItem.Builder()
            .setUri(videoUri)
            .setMimeType("application/mpegts")
            .build()

        val udpDataSourceFactory = DataSource.Factory {
            UdpDataSource(UdpDataSource.DEFAULT_SOCKET_TIMEOUT_MILLIS * 4)
        }
        val mediaSource = ProgressiveMediaSource.Factory(udpDataSourceFactory)
            .createMediaSource(mediaItem)

        exoPlayer?.setMediaSource(mediaSource)
        exoPlayer?.playWhenReady = true

        exoPlayer?.addListener(object : Player.Listener {
            override fun onPlaybackStateChanged(playbackState: Int) {
                val stateString = when (playbackState) {
                    ExoPlayer.STATE_IDLE -> "STATE_IDLE"
                    ExoPlayer.STATE_BUFFERING -> "STATE_BUFFERING"
                    ExoPlayer.STATE_READY -> {
                        if (!isPlayerReady.get()) {
                            isPlayerReady.set(true)
                            Log.i(TAG, "ExoPlayer state changed to: STATE_READY. isPlayerReady is now true.")
                        }
                        "STATE_READY"
                    }
                    ExoPlayer.STATE_ENDED -> "STATE_ENDED"
                    else -> "UNKNOWN_STATE"
                }
                Log.d(TAG, "ExoPlayer state changed to: $stateString")
            }

            override fun onPlayerError(error: PlaybackException) {
                Log.e(TAG, "ExoPlayer Error: ${error.message}", error)
                Log.e(TAG, "Full error details:", error)
                val retryDelay = 5000L
                Log.w(TAG, "ExoPlayer encountered an error. Retrying in ${retryDelay}ms...")
                serviceScope.launch(Dispatchers.Main) {
                    delay(retryDelay)
                    releaseExoPlayer()
                    startExoPlayer()
                }
            }
        })
        exoPlayer?.prepare()
        Log.d(TAG, "ExoPlayer prepared and listening on UDP port $UDP_STREAM_PORT.")
    }

    private fun releaseExoPlayer() {
        exoPlayer?.release()
        exoPlayer = null
        isPlayerReady.set(false)
        Log.d(TAG, "ExoPlayer released.")
    }

    private suspend fun processFramesContinuously() {
        var frameCounter = 0
        Log.d(TAG, "=== STARTING CONTINUOUS LATEST-FRAME PROCESSING ===")
        Log.d(TAG, "===============================================")

        while (serviceScope.isActive) {
            try {
                val frameData = synchronized(frameStack) {
                    if (frameStack.isEmpty()) {
                        null
                    } else {
                        frameStack.pop() // Get latest frame (LIFO)
                    }
                }

                if (frameData == null) {
                    delay(10) // Short delay if no frames available
                    continue
                }

                frameCounter++
                Log.d(TAG, "--- Processing Frame #$frameCounter ---")

                val bitmap = BitmapFactory.decodeByteArray(frameData.imageData, 0, frameData.imageData.size)
                if (bitmap != null) {
                    val signBoundingBox = runModelInference(bitmap)

                    if (signBoundingBox != null) {
                        Log.d(TAG, "✅ App's model detected a sign. Bounding Box (pixels): ${signBoundingBox.contentToString()}")

                        val truckLocation = lastKnownLocation.get()
                        if (truckLocation == null) {
                            Log.w(TAG, "Truck GPS location not yet available. Skipping frame.")
                            bitmap.recycle()
                            continue
                        }

                        val currentTruckLat = truckLocation.latitude
                        val currentTruckLon = truckLocation.longitude

                        val currentBearing = lastKnownBearing.get()?.toDouble() ?: CAMERA_BEARING_DEG

                        val (horizontalDistance, relativeHorizontalAngleToSignDeg) = calculateSignDistanceAndRelativeAngle(
                            focalLengthMm = FOCAL_LENGTH_MM,
                            realObjectHeightM = SIGN_REAL_HEIGHT_M,
                            realObjectWidthM = SIGN_REAL_WIDTH_M,
                            objectPixelHeight = signBoundingBox[3],
                            objectPixelWidth = signBoundingBox[2],
                            sensorHeightMm = SENSOR_HEIGHT_MM,
                            sensorWidthMm = SENSOR_WIDTH_MM,
                            imageResolutionHeightPixels = FRAME_HEIGHT,
                            imageResolutionWidthPixels = FRAME_WIDTH,
                            cameraTiltDeg = CAMERA_TILT_DEG,
                            cameraAltitudeM = CAMERA_ALTITUDE_M,
                            signBoundingBoxPixels = signBoundingBox
                        )

                        val trueBearingToSignDeg = currentBearing + relativeHorizontalAngleToSignDeg
                        val normalizedTrueBearingToSignDeg = (trueBearingToSignDeg % 360 + 360) % 360

                        val (signLat, signLon) = calculateNewGpsPoint(
                            startLat = currentTruckLat,
                            startLon = currentTruckLon,
                            distanceM = horizontalDistance,
                            bearingDeg = normalizedTrueBearingToSignDeg
                        )

                        val uniqueId = generateUniqueId(currentTruckLat, currentTruckLon, horizontalDistance)
                        val signGpsPosition = "%.6f_%.6f".format(signLat, signLon)

                        if (isSignAlreadyTracked(uniqueId, signLat, signLon, horizontalDistance)) {
                            Log.d(TAG, "❌ Sign is already in our local tracking list. Skipping frame transmission to tower.")
                            bitmap.recycle()
                            continue
                        }

                        Log.d(TAG, "Sign is new. Sending to server...")
                        //saveBitmapForDebugging(bitmap, uniqueId)

                        serviceScope.launch(Dispatchers.IO) {
                            try {
                                val signDetectedByServer = processFrameWithProtocol(bitmap, uniqueId, signGpsPosition, signLat, signLon, horizontalDistance)
                                if (signDetectedByServer) {
                                    Log.d(TAG, "Sign detected by server")
                                } else {
                                    Log.d(TAG, "❌ Server did not detect sign or timed out. Continuing to next frame.")
                                }
                            } finally {
                                bitmap.recycle()
                            }
                        }
                    } else {
                        Log.d(TAG, "❌ No sign detected by on-device model. Saving to NoSignDetected folder.")

                        // Save frame without detected sign to NoSignDetected folder
                        serviceScope.launch(Dispatchers.IO) {
                            //saveBitmapToNoSignFolder(bitmap, frameCounter)
                            bitmap.recycle()
                        }
                    }
                } else {
                    Log.w(TAG, "Failed to decode bitmap from frame data")
                }
            } catch (e: InterruptedException) {
                Log.e(TAG, "Frame queue take interrupted. Service is stopping.")
                break
            }
            delay(FRAME_PROCESSING_DELAY)
        }
        Log.d(TAG, "=== CONTINUOUS FRAME PROCESSING STOPPED ===")
    }

    private fun saveBitmapToNoSignFolder(bitmap: Bitmap, frameCounter: Int) {
        try {
            val picturesDir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), NO_SIGN_DETECTED_DIRECTORY)
            if (!picturesDir.exists()) {
                picturesDir.mkdirs()
            }

            val fileName = "no_sign_frame_${frameCounter}_${System.currentTimeMillis()}.jpeg"
            val file = File(picturesDir, fileName)

            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 70, out)
            }
            Log.d(TAG, "DEBUG: Saved frame without sign detection #$frameCounter to ${file.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "ERROR: Failed to save no-sign bitmap for frame #$frameCounter", e)
        }
    }

    private fun saveBitmapForDebugging(bitmap: Bitmap, uniqueId: String) {
        try {
            val picturesDir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "SignDetectionImages")
            if (!picturesDir.exists()) {
                picturesDir.mkdirs()
            }

            val fileName = "sent_image_${uniqueId}.jpeg"
            val file = File(picturesDir, fileName)

            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 70, out)
            }
            Log.d(TAG, "DEBUG: Saved image with ID $uniqueId to ${file.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "ERROR: Failed to save debug bitmap for ID $uniqueId", e)
        }
    }

    private fun isSignAlreadyTracked(currentUniqueId: String, currentSignLat: Double, currentSignLon: Double, currentDistanceToSign: Double): Boolean {
        for (trackedSignRecord in trackedSigns) {
            val results = floatArrayOf(0f, 0f, 0f)
            Location.distanceBetween(
                currentSignLat, currentSignLon,
                trackedSignRecord.signLat, trackedSignRecord.signLon,
                results
            )
            val distanceBetweenSigns = results[0].toDouble()

            if (abs(distanceBetweenSigns) < SAME_SIGN_DISTANCE_EPSILON) {
                Log.d(TAG, "Local check: Found a match. The sign is likely a duplicate.")
                return true
            }
        }
        return false
    }

    private fun addSignToTrackedList(uniqueId: String, signLat: Double, signLon: Double, distanceToSign: Double) {
        if (!isSignAlreadyTracked(uniqueId, signLat, signLon, distanceToSign)) {
            trackedSigns.add(SignRecord(uniqueId, signLat, signLon, distanceToSign))
            Log.d(TAG, "New sign confirmed by tower and added to tracking list: $uniqueId. Total tracked: ${trackedSigns.size}")
        } else {
            Log.d(TAG, "Sign already exists in tracked list, not adding again.")
        }
    }

    private fun runModelInference(bitmap: Bitmap): IntArray? {
        if (tflite == null) {
            Log.e(TAG, "TensorFlow Lite interpreter not initialized.")
            return null
        }

        try {
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(IMAGE_SIZE, IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(NORMALIZE_MEAN, NORMALIZE_STD))
                .build()

            var tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            tensorImage = imageProcessor.process(tensorImage)

            val outputTensorIndex = 0
            val outputShape = tflite!!.getOutputTensor(outputTensorIndex).shape()
            val outputDataType = tflite!!.getOutputTensor(outputTensorIndex).dataType()

            if (outputShape.size != 3 || outputShape[0] != 1 || outputShape[1] != 5 || outputShape[2] != 8400) {
                Log.e(TAG, "Unexpected model output shape: ${outputShape.joinToString()}. Expected [1, 5, 8400]")
                return null
            }

            val outputBuffer = TensorBuffer.createFixedSize(outputShape, outputDataType)
            tflite!!.run(tensorImage.buffer, outputBuffer.buffer)

            // This is where we will process the output, similar to the Python script
            val outputArray = outputBuffer.floatArray

            // The model output is in shape [1, 5, 8400].
            // We need to re-organize it to [8400, 5] to match the Python logic.
            val numDetections = outputShape[2]
            val postprocessedDetections = mutableListOf<FloatArray>()

            for (i in 0 until numDetections) {
                val offset = i
                val confidence = outputArray[4 * numDetections + offset] // Index for confidence

                if (confidence >= DETECTION_THRESHOLD) {
                    val xCenterNormalized = outputArray[0 * numDetections + offset]
                    val yCenterNormalized = outputArray[1 * numDetections + offset]
                    val widthNormalized = outputArray[2 * numDetections + offset]
                    val heightNormalized = outputArray[3 * numDetections + offset]
                    val confidenceScore = outputArray[4 * numDetections + offset]

                    postprocessedDetections.add(
                        floatArrayOf(
                            xCenterNormalized,
                            yCenterNormalized,
                            widthNormalized,
                            heightNormalized,
                            confidenceScore
                        )
                    )
                }
            }

            if (postprocessedDetections.isEmpty()) {
                return null
            }

            // At this point, you would typically implement Non-Maximum Suppression (NMS)
            // This is a crucial step to remove overlapping boxes.
            // NMS is more complex to implement in native Kotlin without a specialized library.
            // A simplified approach might involve just selecting the highest confidence score detection.
            val bestDetection = postprocessedDetections.maxByOrNull { it[4] }
                ?: return null

            val xCenterNormalized = bestDetection[0]
            val yCenterNormalized = bestDetection[1]
            val widthNormalized = bestDetection[2]
            val heightNormalized = bestDetection[3]

            // Scale coordinates back to the original frame size
            val xmin = ((xCenterNormalized - widthNormalized / 2) * FRAME_WIDTH).toInt()
            val ymin = ((yCenterNormalized - heightNormalized / 2) * FRAME_HEIGHT).toInt()
            val xmax = ((xCenterNormalized + widthNormalized / 2) * FRAME_WIDTH).toInt()
            val ymax = ((yCenterNormalized + heightNormalized / 2) * FRAME_HEIGHT).toInt()

            val finalX = xmin.coerceIn(0, FRAME_WIDTH - 1)
            val finalY = ymin.coerceIn(0, FRAME_HEIGHT - 1)
            val finalWidth = (xmax - xmin).coerceIn(1, FRAME_WIDTH - finalX)
            val finalHeight = (ymax - ymin).coerceIn(1, FRAME_HEIGHT - finalY)

            Log.d(TAG, "✅ Detected score: ${bestDetection[4]} (Threshold: $DETECTION_THRESHOLD)")
            return intArrayOf(finalX, finalY, finalWidth, finalHeight)

        } catch (e: Exception) {
            Log.e(TAG, "Error running TFLite inference: ${e.message}", e)
            return null
        }
    }

    private fun generateUniqueId(truckLat: Double, truckLon: Double, distanceToSign: Double): String {
        return "%.6f_%.6f_%.2f".format(truckLat, truckLon, distanceToSign)
    }

    private suspend fun processFrameWithProtocol(bitmap: Bitmap, uniqueId: String, signGpsPosition: String?, signLat: Double, signLon: Double, distanceToSign: Double): Boolean {
        if (udpSender == null) {
            Log.e(TAG, "UdpSender is not initialized. Cannot send image.")
            return false
        }

        try {
            val jpegBytes = ImageConverter.bitmapToJpegBytes(bitmap)
            val responseString = udpSender!!.sendImageAndGetResponse(jpegBytes, uniqueId, signGpsPosition)

            if (responseString.startsWith("OK:")) {
                val responseJson = responseString.substring(3)
                try {
                    val response = gson.fromJson(responseJson, ResponseForSignDetection::class.java)
                    if (response.IsSignDetected) {
                        Log.d(TAG, "🎯 SIGN DETECTED by Server! (ID: ${response.UniqueSequencedID})")
                        addSignToTrackedList(uniqueId, signLat, signLon, distanceToSign)
                        return true
                    } else {
                        Log.d(TAG, "❌ Server did not detect sign (ID: ${response.UniqueSequencedID}).")
                        return false
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error parsing server response JSON for ID: $uniqueId, data: '$responseJson'", e)
                    return false
                }
            } else {
                Log.e(TAG, "Server communication error or timeout for ID: $uniqueId. Response: $responseString")
                return false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in protocol communication (ID: $uniqueId)", e)
            return false
        }
    }

    private fun startForegroundService() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                "Image UDP Service Channel",
                NotificationManager.IMPORTANCE_DEFAULT
            )
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }

        val notification: Notification = NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle("Sign Detection Active")
            .setContentText("Processing images for sign detection...")
            .setSmallIcon(R.mipmap.ic_launcher)
            .build()

        startForeground(NOTIFICATION_ID, notification)
    }

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }

    override fun onDestroy() {
        super.onDestroy()
        serviceJob.cancel()
        handlerThread?.quitSafely()
        exoPlayer?.release()
        exoPlayer = null
        imageReader?.close()
        imageReader = null
        tflite?.close()

        synchronized(frameStack) {
            frameStack.clear()
        }

        serviceLocationManager?.let { manager ->
            serviceLocationListener?.let { listener ->
                manager.removeUpdates(listener)
                Log.d(TAG, "Service: Stopped LocationManager updates during onDestroy.")
            }
        }

        udpSender?.close()

        Log.d(TAG, "Service destroyed.")
    }

    private fun calculateSignDistanceAndRelativeAngle(
        focalLengthMm: Double,
        realObjectHeightM: Double,
        realObjectWidthM: Double,
        objectPixelHeight: Int,
        objectPixelWidth: Int,
        sensorHeightMm: Double,
        sensorWidthMm: Double,
        imageResolutionWidthPixels: Int,
        imageResolutionHeightPixels: Int,
        cameraTiltDeg: Double,
        cameraAltitudeM: Double,
        signBoundingBoxPixels: IntArray
    ): Pair<Double, Double> {
        var currentFocalLengthMm = focalLengthMm

        val signCenterX = signBoundingBoxPixels[0].toDouble() + signBoundingBoxPixels[2].toDouble() / 2
        val edgeThresholdLeft = 100.0
        val edgeThresholdRight = imageResolutionWidthPixels - 100.0

        if (signCenterX < edgeThresholdLeft || signCenterX > edgeThresholdRight) {
            val FIXED_FOCAL_LENGTH_FOR_EDGE_DETECTION = 3.8
            println("Sign detected near image edge (X-axis). Increasing focal length to $FIXED_FOCAL_LENGTH_FOR_EDGE_DETECTION mm.")
            currentFocalLengthMm = FIXED_FOCAL_LENGTH_FOR_EDGE_DETECTION
        }

        val objectSensorHeightMm = (objectPixelHeight.toDouble() / imageResolutionHeightPixels) * sensorHeightMm
        val objectSensorWidthMm = (objectPixelWidth.toDouble() / imageResolutionWidthPixels) * sensorWidthMm

        var distanceToObjectHeightM: Double
        if (objectSensorHeightMm == 0.0) {
            println("Warning: Object pixel height or sensor height is zero, cannot calculate distance by height.")
            distanceToObjectHeightM = Double.POSITIVE_INFINITY
        } else {
            distanceToObjectHeightM = (currentFocalLengthMm * (realObjectHeightM * 1000)) / objectSensorHeightMm / 1000
        }

        var distanceToObjectWidthM: Double
        if (objectSensorWidthMm == 0.0) {
            println("Warning: Object pixel width or sensor width is zero, cannot calculate distance by width.")
            distanceToObjectWidthM = Double.POSITIVE_INFINITY
        } else {
            distanceToObjectWidthM = (currentFocalLengthMm * (realObjectWidthM * 1000)) / objectSensorWidthMm / 1000
        }

        val straightLineDistanceM: Double

        val avgPixelDim = (objectPixelHeight + objectPixelWidth) / 2.0
        val pixelDimDifferencePercentage = if (avgPixelDim > 0) {
            (abs(objectPixelWidth - objectPixelHeight) / avgPixelDim) * 100
        } else {
            0.0
        }

        val THRESHOLD_PERCENTAGE = 80.0

        if (pixelDimDifferencePercentage > THRESHOLD_PERCENTAGE) {
            println("Pixel dimensions differ by >${THRESHOLD_PERCENTAGE}%. Averaging height and width distances.")
            if (distanceToObjectHeightM.isInfinite() && distanceToObjectWidthM.isInfinite()) {
                straightLineDistanceM = Double.POSITIVE_INFINITY
            } else if (distanceToObjectHeightM.isInfinite()) {
                straightLineDistanceM = distanceToObjectWidthM
            } else if (distanceToObjectWidthM.isInfinite()) {
                straightLineDistanceM = distanceToObjectHeightM
            } else {
                straightLineDistanceM = (distanceToObjectHeightM + distanceToObjectWidthM) / 2
            }
        } else {
            println("Pixel dimensions differ by <=${THRESHOLD_PERCENTAGE}%. Using only height-based distance.")
            straightLineDistanceM = distanceToObjectHeightM
        }

        if (straightLineDistanceM.isInfinite()) {
            return Pair(Double.POSITIVE_INFINITY, 0.0)
        }

        val centerPixelY = signBoundingBoxPixels[1].toDouble() + signBoundingBoxPixels[3].toDouble() / 2
        val centerYOffsetPixels = centerPixelY - (imageResolutionHeightPixels.toDouble() / 2)
        val centerYOffsetMm = centerYOffsetPixels * (sensorHeightMm / imageResolutionHeightPixels)
        val angleFromOpticalAxisToCenterYRad = atan2(centerYOffsetMm, currentFocalLengthMm)
        val absoluteAngleToCenterRad = Math.toRadians(cameraTiltDeg) + angleFromOpticalAxisToCenterYRad

        var horizontalDistanceM: Double
        val cosAngle = cos(absoluteAngleToCenterRad)
        if (abs(cosAngle) < 1e-6) {
            println("Warning: Object is nearly directly above/below the camera, horizontal distance might be very small or undefined.")
            horizontalDistanceM = 0.0
        } else {
            horizontalDistanceM = straightLineDistanceM * cosAngle
            horizontalDistanceM = abs(horizontalDistanceM)
        }

        val centerPixelX = signBoundingBoxPixels[0].toDouble() + signBoundingBoxPixels[2].toDouble() / 2
        val centerXOffsetPixels = centerPixelX - (imageResolutionWidthPixels.toDouble() / 2)
        val centerXOffsetMm = centerXOffsetPixels * (sensorWidthMm / imageResolutionWidthPixels)
        val horizontalAngleToSignRad = atan2(centerXOffsetMm, currentFocalLengthMm)
        val horizontalAngleToSignDeg = Math.toDegrees(horizontalAngleToSignRad)

        return Pair(horizontalDistanceM, horizontalAngleToSignDeg)
    }

    private fun calculateNewGpsPoint(
        startLat: Double,
        startLon: Double,
        distanceM: Double,
        bearingDeg: Double
    ): Pair<Double, Double> {
        val earthRadius = 6371000.0
        val latRad = Math.toRadians(startLat)
        val lonRad = Math.toRadians(startLon)
        val bearingRad = Math.toRadians(bearingDeg)

        val newLatRad = asin(sin(latRad) * cos(distanceM / earthRadius) +
                cos(latRad) * sin(distanceM / earthRadius) * cos(bearingRad))

        val newLonRad = lonRad + atan2(sin(bearingRad) * sin(distanceM / earthRadius) * cos(latRad),
            cos(distanceM / earthRadius) - sin(latRad) * sin(newLatRad))

        return Pair(Math.toDegrees(newLatRad), Math.toDegrees(newLonRad))
    }
}