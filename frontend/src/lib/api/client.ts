import axios, { AxiosError, AxiosInstance } from 'axios'

// API 响应类型
export interface ApiResponse<T> {
  success: boolean
  data: T | null
  error: string | null
  message: string | null
}

// 创建 axios 实例
const apiClient: AxiosInstance = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<ApiResponse<unknown>>) => {
    const message = error.response?.data?.error || error.message || 'Unknown error'
    return Promise.reject(new Error(message))
  }
)

export { apiClient }
