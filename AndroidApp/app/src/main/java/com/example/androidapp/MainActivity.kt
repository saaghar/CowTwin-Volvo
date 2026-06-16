package com.example.androidapp // Your MainActivity's package

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.androidapp.LocationStorage // Corrected import for LocationStorage
import com.example.androidapp.LocationStorage.getLocation
import com.volvogroup.cargoliftapp.ui.components.checkPermissions // Corrected import for checkPermissions
import com.example.androidapp.ui.theme.AndroidAppTheme // Assuming you have a default theme
import com.volvogroup.cargoliftapp.ui.components.startLocationUpdates

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private val LOCATION_PERMISSION_REQUEST_CODE = 1001

    @RequiresApi(Build.VERSION_CODES.S)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Request permissions immediately
        // checkPermissions() now handles this internally and calls startLocationUpdates()
        // if permissions are already granted.
        checkPermissions()


        setContent {
            AndroidAppTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    AppScreen(activityContext = this)
                }
            }
        }
    }

    @RequiresApi(Build.VERSION_CODES.S)
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == LOCATION_PERMISSION_REQUEST_CODE && grantResults.isNotEmpty()
            && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startLocationUpdates() // Call startLocationUpdates if permission granted
            Toast.makeText(this, "Location permission granted.", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "Location permission denied. Service functionality may be limited.", Toast.LENGTH_LONG).show()
            Log.w("MainActivity", "Location permission denied.")
        }
    }
}

// Composables for the UI
@RequiresApi(Build.VERSION_CODES.S) // Keep if getLocation still has S requirement
@Composable
fun AppScreen(activityContext: MainActivity) { // Pass the activity context
    // State to hold the current location string, updates whenever LocationStorage changes
    var currentLocationText by remember { mutableStateOf("Location: Not Available") }

    // Use DisposableEffect to start and stop a periodic update for the location display
    DisposableEffect(Unit) {
        val job = CoroutineScope(Dispatchers.Main).launch {
            while (isActive) {
                // Correctly retrieve location from LocationStorage
                currentLocationText = "Location: ${LocationStorage.getLocationString()}"
                delay(1000) // Update every second
            }
        }
        onDispose {
            job.cancel() // Cancel the coroutine when the composable leaves the composition
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // --- ADDED BUTTONS BACK HERE ---
        Button(onClick = {
            // Start the ImageUdpService
            val serviceIntent = Intent(activityContext, ImageUdpService::class.java)
            // Use startForegroundService for services that perform long-running tasks
            ContextCompat.startForegroundService(activityContext, serviceIntent)
            Toast.makeText(activityContext, "ImageUdpService Started", Toast.LENGTH_SHORT).show()
        }) {
            Text("Start Service")
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = {
            // Stop the ImageUdpService
            val serviceIntent = Intent(activityContext, ImageUdpService::class.java)
            activityContext.stopService(serviceIntent)
            Toast.makeText(activityContext, "ImageUdpService Stopped", Toast.LENGTH_SHORT).show()
        }) {
            Text("Stop Service")
        }

        Spacer(modifier = Modifier.height(32.dp))
        // --- END ADDED BUTTONS ---

        Text(text = currentLocationText)
    }
}

// Optional: Preview function for Android Studio design view
@Preview(showBackground = true)
@Composable
@RequiresApi(Build.VERSION_CODES.S)
fun DefaultPreview() {
    AndroidAppTheme {
        // For preview, we pass a dummy ComponentActivity or provide mock context
        // Note: This preview won't actually start/stop services as it's not a real activity context.
        AppScreen(activityContext = MainActivity())
    }
}