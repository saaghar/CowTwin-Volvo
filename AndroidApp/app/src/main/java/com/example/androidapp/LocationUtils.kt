package com.volvogroup.cargoliftapp.ui.components // This is the package you provided for these utilities

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context.LOCATION_SERVICE
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.androidapp.MainActivity // Ensure this import is correct based on your MainActivity's package
import com.example.androidapp.LocationStorage // Import your LocationStorage singleton

/**
 * Checks the permission to get the location of the device.
 * Requests permission if not granted.
 */
private val LOCATION_PERMISSION_REQUEST_CODE = 1001

fun MainActivity.checkPermissions() {
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
        != PackageManager.PERMISSION_GRANTED)
    {
        Log.d("Permissions", "Location permission NOT granted, requesting...")
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION), // Request both
            LOCATION_PERMISSION_REQUEST_CODE
        )
    }
    else{
        Log.d("Permissions", "Location permission ALREADY granted.")
        startLocationUpdates()
    }
}

/**
 * Retrieves the last known location using GPS or Network provider and stores it in LocationStorage.
 * Requires ACCESS_FINE_LOCATION permission.
 */
@SuppressLint("MissingPermission")
fun MainActivity.startLocationUpdates() {
    val locationManager = getSystemService(LOCATION_SERVICE) as LocationManager

    val locationListener = object : LocationListener {
        override fun onLocationChanged(location: Location) {
            val latitude = location.latitude
            val longitude = location.longitude

            LocationStorage.setLocation(latitude, longitude)
            // You can update UI or do something here with new location
        }

        override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
        override fun onProviderEnabled(provider: String) {}
        override fun onProviderDisabled(provider: String) {}
    }

    val isGPSEnabled = locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)
    val isNetworkEnabled = locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)

    val provider = if (isGPSEnabled) LocationManager.GPS_PROVIDER else if (isNetworkEnabled) LocationManager.NETWORK_PROVIDER else null

    provider?.let {
        locationManager.requestLocationUpdates(
            it,
            1000L, // minimum time interval between updates in milliseconds (1 second)
            1f,    // minimum distance between updates in meters
            locationListener
        )
    }
}