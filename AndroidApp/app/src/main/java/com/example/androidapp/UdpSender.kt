package com.example.androidapp

import android.util.Log
import com.google.gson.Gson
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.io.IOException
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.SocketTimeoutException
import java.util.concurrent.ConcurrentHashMap

// A data class to hold the pending request's deferred response
data class PendingRequest(
    val deferred: CompletableDeferred<String>
)

class UdpSender(
    private val serverIp: String,
    private val serverPort: Int,
    private val TAG: String,
    private val gson: Gson
) {
    private var udpSocket: DatagramSocket? = null
    private var serverAddress: InetAddress? = null

    // Coroutine scope for managing background tasks like the response listener
    private val senderScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    // A thread-safe map to store requests waiting for a response
    private val pendingRequests = ConcurrentHashMap<String, PendingRequest>()

    // Initialize the socket and address, and start the response listener
    fun initialize() {
        try {
            udpSocket = DatagramSocket()
            serverAddress = InetAddress.getByName(serverIp)
            Log.d(TAG, "UdpSender initialized: socket created, server address resolved.")

            // Start the coroutine that listens for responses
            senderScope.launch {
                listenForResponses()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize UdpSender: ${e.message}", e)
            udpSocket?.close()
            udpSocket = null
            serverAddress = null
        }
    }

    // Close the socket and cancel all background tasks
    fun close() {
        senderScope.coroutineContext.cancel()
        udpSocket?.close()
        Log.d(TAG, "UdpSender socket closed and coroutine scope cancelled.")
    }

    // Main function to send an image and wait for a response
    suspend fun sendImageAndGetResponse(jpegBytes: ByteArray, uniqueId: String, signGpsPosition: String?): String {
        return withContext(Dispatchers.IO) {
            if (udpSocket == null || serverAddress == null) {
                return@withContext "ERROR: UdpSender not initialized."
            }

            // Create a deferred object that will be completed when the response arrives
            val deferredResponse = CompletableDeferred<String>()
            pendingRequests[uniqueId] = PendingRequest(deferredResponse)

            try {
                // Fragment the image
                val maxUdpFragmentSize = 1400 // Safe UDP packet size
                val fragments = ImageConverter.splitImageIntoFragments(jpegBytes, maxUdpFragmentSize)

                // 1. Send the initial request message
                val requestMessage = RequestForSignDetection(fragments.size, uniqueId, signGpsPosition)
                val requestJson = gson.toJson(requestMessage)
                val requestData = requestJson.toByteArray(Charsets.UTF_8)
                val requestPacket = DatagramPacket(requestData, requestData.size, serverAddress, serverPort)
                udpSocket!!.send(requestPacket)
                Log.i(TAG, "Sent initial request for $uniqueId. Expecting ${fragments.size} fragments.")

                // 2. Send all fragments
                for ((index, base64Fragment) in fragments.withIndex()) {
                    val fragmentMessage = SendImageMessage(uniqueId, index, base64Fragment)
                    val fragmentJson = gson.toJson(fragmentMessage)
                    val fragmentData = fragmentJson.toByteArray(Charsets.UTF_8)
                    val fragmentPacket = DatagramPacket(fragmentData, fragmentData.size, serverAddress, serverPort)
                    udpSocket!!.send(fragmentPacket)
                    Log.d(TAG, "Sent fragment $index/$uniqueId.")
                    kotlinx.coroutines.delay(10) // Use coroutine delay
                }

                // 3. Wait for the response with a timeout
                Log.i(TAG, "All fragments for $uniqueId sent. Waiting for server response...")
                val responseString = withTimeout(5000L) {
                    deferredResponse.await()
                }

                // Log the final response and return it with the "OK:" prefix
                Log.i(TAG, "Server response for $uniqueId: $responseString")
                "OK:$responseString"

            } catch (e: TimeoutCancellationException) {
                Log.e(TAG, "Server response for $uniqueId timed out after 5 seconds.", e)
                "TIMEOUT"
            } catch (e: Exception) {
                Log.e(TAG, "Error while sending image $uniqueId: ${e.message}", e)
                "NETWORK_ERROR"
            } finally {
                // IMPORTANT: Remove the pending request from the map
                pendingRequests.remove(uniqueId)
            }
        }
    }

    // New function to listen for and process UDP responses
    private suspend fun listenForResponses() {
        // We set a receive timeout here to prevent the coroutine from blocking forever
        // if no data is received. It allows the loop to check for cancellation.
        udpSocket?.soTimeout = 1000

        val responseBuffer = ByteArray(1024)
        val responsePacket = DatagramPacket(responseBuffer, responseBuffer.size)

        while (senderScope.coroutineContext.isActive) {
            try {
                udpSocket?.receive(responsePacket)
                val responseData = responsePacket.data.copyOfRange(0, responsePacket.length)
                val responseString = String(responseData, Charsets.UTF_8)

                // Parse the UniqueSequencedID from the response
                val response = gson.fromJson(responseString, ResponseForSignDetection::class.java)
                val uniqueIdFromResponse = response.UniqueSequencedID

                // Find and complete the matching deferred object
                val pendingRequest = pendingRequests[uniqueIdFromResponse]
                if (pendingRequest != null) {
                    // Log before completing the deferred to capture the raw response
                    Log.d(TAG, "Received response for ID $uniqueIdFromResponse")
                    pendingRequest.deferred.complete(responseString)
                } else {
                    Log.w(TAG, "Received response for an unrecognized or already processed ID: $uniqueIdFromResponse")
                }
            } catch (e: SocketTimeoutException) {
                // This is expected and normal, as the socket timeout is set to 1 second
                continue
            } catch (e: Exception) {
                Log.e(TAG, "Error parsing or processing response: ${e.message}", e)
            }
        }
        Log.d(TAG, "Response listener coroutine stopped.")
    }
}