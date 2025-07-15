# Manual de Usuario - Simulador de Sistemas de Colas

## Índice
1. [Introducción](#introducción)
2. [Requisitos del Sistema](#requisitos-del-sistema)
3. [Instalación y Configuración](#instalación-y-configuración)
4. [Modelos Disponibles](#modelos-disponibles)
5. [Guía de Uso por Modelo](#guía-de-uso-por-modelo)
6. [Formato de Archivos](#formato-de-archivos)
7. [Interpretación de Resultados](#interpretación-de-resultados)
8. [Ejemplos Prácticos](#ejemplos-prácticos)
9. [Solución de Problemas](#solución-de-problemas)
10. [Referencias y Fundamentos Teóricos](#referencias-y-fundamentos-teóricos)

---

## Introducción

Este simulador implementa diferentes modelos de sistemas de colas estocásticos utilizando simulación de eventos discretos. La herramienta permite analizar el desempeño de sistemas de servicio bajo diferentes configuraciones y validar modelos teóricos con datos empíricos.

### Características Principales
- **5 modelos diferentes**: M/M/1, M/M/c/c, M/M/c, Geom/Geom/N/N y validación empírica
- **Generador de números aleatorios robusto**: Implementación lcgrand.cpp
- **Múltiples métricas de desempeño**: Tiempo de espera, utilización, probabilidades de bloqueo
- **Validación estadística**: Comparación teoría vs simulación
- **Interfaz simple**: Archivos de texto para entrada y salida

---

## Requisitos del Sistema

### Hardware Mínimo
- Procesador: 1 GHz o superior
- RAM: 512 MB
- Espacio en disco: 50 MB

### Software
- **Compilador C++**: GCC, MinGW, Visual Studio, o compatible
- **Sistema Operativo**: Windows, Linux, macOS
- **Editor de texto**: Para modificar archivos de parámetros

---

## Instalación y Configuración

### Paso 1: Compilación
```bash
# Para modelo M/M/1 básico
g++ -o Simulador "Sistema de Colas.cpp"

# Para Erlang B
g++ -o SimuladorB "Punto D/SimulacionErlangB.cpp"

# Para Erlang C
g++ -o SimuladorC "Punto D/SimulacionErlangC.cpp"

# Para modelo con datos empíricos
g++ -o SimuladorC "Punto C/Sistema de Colas_Punto_C.cpp"

# Para modelo Geom/Geom/N/N
g++ -o SimuladorE "Punto E/Sistema de Colas.cpp"
```

### Paso 2: Verificación
Ejecute cualquier simulador para verificar que no hay errores de compilación.

---

## Modelos Disponibles

| Modelo | Archivo Principal | Descripción | Aplicaciones |
|--------|-------------------|-------------|--------------|
| **M/M/1** | `Sistema de Colas.cpp` | Cola simple con un servidor | Análisis básico, validación |
| **M/M/c/c (Erlang B)** | `SimulacionErlangB.cpp` | Múltiples servidores, pérdida de clientes | Telecomunicaciones, call centers |
| **M/M/c (Erlang C)** | `SimulacionErlangC.cpp` | Múltiples servidores, cola infinita | Centros de servicio, soporte técnico |
| **Geom/Geom/N/N** | `Punto E/Sistema de Colas.cpp` | Tiempo discreto, capacidad limitada | Sistemas digitales, redes |
| **Validación Empírica** | `Punto C/Sistema de Colas_Punto_C.cpp` | Datos reales vs modelo teórico | Calibración, validación |

---

## Guía de Uso por Modelo

### Modelo M/M/1 Básico

#### Paso 1: Configurar Parámetros
Edite el archivo `param.txt`:
```
8.702287928    # Tiempo promedio entre llegadas (minutos)
5.157664924    # Tiempo promedio de servicio (minutos)
1000           # Número de clientes a simular
```

#### Paso 2: Ejecutar
```bash
./Simulador
```

#### Paso 3: Revisar Resultados
Los resultados se guardan en `result.txt`.

---

### Modelo Erlang B (M/M/c/c)

#### Configuración
Archivo `Punto D/params.txt`:
```
8.702287928    # Tiempo promedio entre llegadas
5.157664924    # Tiempo promedio de servicio
1000           # Número de clientes
4              # Número de servidores
```

#### Ejecución
```bash
cd "Punto D"
./SimuladorB
```

#### Métricas Clave
- **Probabilidad de pérdida (B de Erlang)**: Fracción de clientes perdidos
- **Utilización del servidor**: Fracción de tiempo ocupado

---

### Modelo Erlang C (M/M/c)

#### Configuración
Mismo archivo `params.txt` que Erlang B.

#### Ejecución
```bash
cd "Punto D"
./SimuladorC
```

#### Métricas Clave
- **Probabilidad de espera (C de Erlang)**: Fracción de clientes que esperan
- **Longitud promedio de cola**: Número promedio en espera

---

### Modelo Geom/Geom/N/N (Tiempo Discreto)

#### Parámetros (en código)
```cpp
int N = 5;               // Capacidad del sistema
int m = 5;               // Número de servidores  
float p = 0.8;           // Probabilidad de llegada por slot
float s = 0.4;           // Probabilidad de servicio por slot
int num_slots = 1000000; // Slots a simular
```

#### Ejecución
```bash
cd "Punto E"
./SimuladorE
```

#### Resultados
- **Distribución de estados**: P₀, P₁, ..., P₅
- **Probabilidad de bloqueo**: Fracción de llegadas rechazadas

---

### Validación con Datos Empíricos

#### Preparar Datos
1. **Archivo de parámetros**: `Punto C/param_Punto_C.txt`
2. **Datos empíricos**: `Punto C/datos_Punto_C.tsv` (formato tab-separated)

#### Formato de Datos Empíricos
```
tiempo_llegada_1	tiempo_servicio_1
tiempo_llegada_2	tiempo_servicio_2
...
```

#### Ejecución
```bash
cd "Punto C"
./SimuladorC
```

---

## Formato de Archivos

### Archivos de Entrada

#### param.txt (Modelos básicos)
```
[tiempo_entre_llegadas]
[tiempo_de_servicio] 
[numero_de_clientes]
```

#### params.txt (Modelos multi-servidor)
```
[tiempo_entre_llegadas]
[tiempo_de_servicio]
[numero_de_clientes]
[numero_de_servidores]
```

#### datos_Punto_C.tsv (Validación empírica)
```
[tiempo_llegada_1]	[tiempo_servicio_1]
[tiempo_llegada_2]	[tiempo_servicio_2]
...
```

### Archivos de Salida

#### result.txt (M/M/1)
```
Sistema de Colas Simple

Tiempo promedio de llegada      8.702 minutos
Tiempo promedio de atencion     5.158 minutos
Numero de clientes              1000

Espera promedio en la cola      5.309 minutos
Numero promedio en cola         0.585
Uso del servidor               0.529
Tiempo de terminacion          9078.085 minutos
```

#### resultsB.txt (Erlang B)
```
Sistema de Colas M/M/4/4
Clientes perdidos:              604
Probabilidad de pérdida:        0.377
Uso del servidor:              0.382
```

#### resultsC.txt (Erlang C)
```
Sistema de Colas M/M/4
Clientes que esperaron:         117
Probabilidad de espera:         0.117
Longitud promedio de la cola:   0.173
```

---

## Interpretación de Resultados

### Métricas Básicas (M/M/1)

| Métrica | Fórmula Teórica | Interpretación |
|---------|-----------------|----------------|
| **Utilización (ρ)** | λ/μ | Fracción de tiempo que el servidor está ocupado |
| **Número promedio en cola (Lq)** | ρ²/(1-ρ) | Clientes esperando en promedio |
| **Tiempo promedio en cola (Wq)** | ρ/(μ(1-ρ)) | Tiempo promedio de espera |
| **Número promedio en sistema (L)** | ρ/(1-ρ) | Clientes en el sistema (cola + servicio) |

### Validación de Resultados

#### Verificar Estabilidad
- **ρ < 1**: Sistema estable
- **ρ ≥ 1**: Sistema inestable (cola crece indefinidamente)

#### Leyes de Little
- **L = λ × W**: Número promedio = Tasa × Tiempo promedio
- **Lq = λ × Wq**: Verificar consistencia

#### Comparación Teórica vs Simulación
```
Error relativo = |Valor_Simulado - Valor_Teórico| / Valor_Teórico × 100%
```
- **< 5%**: Excelente aproximación
- **5-10%**: Buena aproximación  
- **> 10%**: Revisar parámetros o aumentar muestra

---

## Ejemplos Prácticos

### Ejemplo 1: Análisis de un Banco

**Problema**: Un banco tiene un cajero que atiende clientes cada 5 minutos en promedio. Los clientes llegan cada 8 minutos en promedio.

**Configuración** (`param.txt`):
```
8.0    # 8 minutos entre llegadas
5.0    # 5 minutos de servicio
1000   # 1000 clientes
```

**Análisis**:
- ρ = 5/8 = 0.625 (sistema estable)
- Tiempo promedio en cola esperado: 0.625/(1-0.625) × 5 = 8.33 minutos

### Ejemplo 2: Call Center con Múltiples Agentes

**Problema**: Call center con 4 agentes, llamadas cada 2 minutos, atención promedio 7 minutos.

**Configuración** (`params.txt`):
```
2.0    # 2 minutos entre llamadas
7.0    # 7 minutos de atención
1000   # 1000 llamadas
4      # 4 agentes
```

**Análisis**:
- λ = 0.5 llamadas/min, μ = 1/7 ≈ 0.143 servicios/min por agente
- ρ = 0.5/(4×0.143) = 0.875 (sistema estable)

### Ejemplo 3: Sistema de Red Digital

**Problema**: Router con 5 puertos, probabilidad de llegada 0.8, probabilidad de transmisión 0.4.

**Modificar en código** (`Punto E/Sistema de Colas.cpp`):
```cpp
int N = 5;      // 5 puertos
int m = 5;      // 5 canales de transmisión
float p = 0.8;  // Probabilidad de llegada
float s = 0.4;  // Probabilidad de transmisión
```

---

## Solución de Problemas

### Problemas Comunes

#### Error de Compilación
```
error: 'lcgrand' was not declared
```
**Solución**: Verificar que `lcgrand.cpp` esté en el mismo directorio.

#### Archivo no encontrado
```
Error: No se puede abrir param.txt
```
**Solución**: 
1. Verificar que el archivo existe
2. Verificar permisos de lectura
3. Ejecutar desde el directorio correcto

#### Resultados Inconsistentes
**Posibles Causas**:
1. **Sistema inestable** (ρ ≥ 1): Reducir λ o aumentar μ
2. **Muestra pequeña**: Aumentar número de clientes
3. **Período de calentamiento**: Los primeros eventos pueden sesgar resultados

#### Simulación muy lenta
**Soluciones**:
1. Reducir número de clientes/slots
2. Optimizar compilación: `g++ -O2 -o simulador archivo.cpp`
3. Usar datos más pequeños para pruebas

### Validación de Entrada

#### Verificar Parámetros
```cpp
// Añadir validaciones
if (media_entre_llegadas <= 0 || media_atencion <= 0) {
    printf("Error: Los tiempos deben ser positivos\n");
    return -1;
}

if (num_esperas_requerido <= 0) {
    printf("Error: Número de clientes debe ser positivo\n");
    return -1;
}
```

### Debugging

#### Verificar Convergencia
```cpp
// Imprimir estadísticas intermedias cada 100 clientes
if (num_clientes_espera % 100 == 0) {
    printf("Cliente %d: Util=%.3f, Cola=%.3f\n", 
           num_clientes_espera,
           area_estado_servidor/tiempo_simulacion,
           area_num_entra_cola/tiempo_simulacion);
}
```

---

## Referencias y Fundamentos Teóricos

### Teoría de Colas
- **Kendall Notation**: A/B/c/d/e/f
  - A: Proceso de llegadas
  - B: Proceso de servicio  
  - c: Número de servidores
  - d: Capacidad del sistema
  - e: Población
  - f: Disciplina de cola

### Distribuciones Utilizadas

#### Exponencial
- **Función de densidad**: f(x) = λe^(-λx)
- **Media**: 1/λ
- **Propiedad**: Memoryless

#### Geométrica (Tiempo Discreto)
- **Función de masa**: P(X=k) = (1-p)^(k-1) × p
- **Media**: 1/p

### Fórmulas Importantes

#### M/M/1
```
ρ = λ/μ
π₀ = 1 - ρ
πₙ = ρⁿ(1-ρ)
L = ρ/(1-ρ)
W = 1/(μ-λ)
Lq = ρ²/(1-ρ)
Wq = ρ/(μ(1-ρ))
```

#### Erlang B (M/M/c/c)
```
B(c,A) = (A^c/c!) / Σ(i=0 to c)(A^i/i!)
```

#### Erlang C (M/M/c)
```
C(c,A) = B(c,A) / (1 - A/c + B(c,A))
```

### Simulación de Eventos Discretos
1. **Inicialización**: Estado inicial del sistema
2. **Control de tiempo**: Avanzar al próximo evento
3. **Procesamiento**: Ejecutar lógica del evento
4. **Actualización**: Modificar estado y estadísticas
5. **Repetición**: Hasta condición de parada

---

## Contacto y Soporte

Para preguntas adicionales o reportar problemas:
- Revisar la documentación en `READ ME.txt`
- Verificar implementación en archivos fuente
- Consultar literatura especializada en teoría de colas

---

**Versión**: 1.0  
**Fecha**: Julio 2025  
**Autor**: Proyecto Estocásticos Taller 2
