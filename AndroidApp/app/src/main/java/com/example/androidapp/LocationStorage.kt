package com.example.androidapp // Make sure this package matches your app's base package

import java.util.concurrent.atomic.AtomicReference

/**
 * Singleton object to store and retrieve the latest known GPS location.
 * Uses AtomicReference for thread-safe access to the location data.
 */
object LocationStorage {
    // AtomicReference is used to safely update and retrieve the Pair<latitude, longitude>
    // across different threads (e.g., UI thread updating, service thread reading).
    private val _currentLocation = AtomicReference<Pair<Double, Double>?>(null)

    /**
     * Sets the current latitude and longitude.
     * @param lat The latitude.
     * @param lon The longitude.
     */
    fun setLocation(lat: Double, lon: Double) {
        _currentLocation.set(Pair(lat, lon))
    }

    /**
     * Retrieves the current latitude and longitude.
     * @return A Pair of latitude and longitude (Double, Double), or null if not available.
     */
    fun getLocation(): Pair<Double, Double>? {
        return _currentLocation.get()
    }

    /**
     * Returns a formatted string of the current location for debugging and UI display.
     */
    fun getLocationString(): String {
        val loc = _currentLocation.get()
        return if (loc != null) {
            "Lat: %.6f, Lon: %.6f".format(loc.first, loc.second)
        } else {
            "Not Available"
        }
    }
}