class AnglePID:
    def __init__(self, Kp, Ki, Kd, setpoint, v_max, a_max, dt):
        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd
        self._setpoint = setpoint
        self._v_max = v_max
        self._a_max = a_max
        self._error_last = 0
        self._integral = 0
        self._dt = dt
        self._v_current = 0

    def update_setpoint(self, setpoint):
        self._setpoint = setpoint

    def update(self, p_current):
        error = self._setpoint - p_current
        self._integral = (self._integral + error) * self._dt
        derivative = (error - self._error_last) / self._dt
        pid_value = self._Kp * error + self._Ki * self._integral + self._Kd * derivative

        a_current = max(-self._a_max, min(self._a_max, pid_value))
        self._v_current = max(-self._v_max, min(self._v_max, self._v_current + a_current * self._dt))
        p_next = p_current + self._v_current * self._dt

        self._error_last = error

        return p_next, a_current, self._v_current