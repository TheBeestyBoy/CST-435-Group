import { useState, useEffect } from 'react'
import { FileText, AlertCircle } from 'lucide-react'

export default function Resume() {
  const [hasResume, setHasResume] = useState(false)
  const [resumeFile, setResumeFile] = useState(null)

  useEffect(() => {
    // Check if resume PDF exists in public/resume folder
    fetch('/resume/')
      .then(res => {
        if (res.ok) {
          // Try to find any PDF file
          // Since we can't list directory contents from frontend, we'll check common filenames
          checkForPDF()
        }
      })
      .catch(() => {
        setHasResume(false)
      })
  }, [])

  const checkForPDF = () => {
    // Try common resume filenames
    const commonNames = ['resume.pdf', 'Resume.pdf', 'RESUME.pdf', 'resume', 'Resume', 'RESUME']

    const checkFile = (filename) => {
      return fetch(`/resume/${filename}`)
        .then(res => {
          if (res.ok) {
            setHasResume(true)
            setResumeFile(`/resume/${filename}`)
            return true
          }
          return false
        })
        .catch(() => false)
    }

    // Check files in sequence
    let found = false
    for (const name of commonNames) {
      if (!found) {
        checkFile(name).then(result => {
          if (result) found = true
        })
      }
    }
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center space-x-3 mb-8">
          <FileText size={32} className="text-blue-600" />
          <h1 className="text-3xl font-bold text-gray-800">Resume</h1>
        </div>

        {!hasResume ? (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-8 text-center">
            <AlertCircle size={48} className="mx-auto text-yellow-600 mb-4" />
            <h2 className="text-xl font-semibold text-gray-800 mb-2">No Resume Found</h2>
            <p className="text-gray-600 mb-4">
              To add your resume, simply drag and drop a PDF file into the{' '}
              <code className="bg-gray-200 px-2 py-1 rounded font-mono">launcher/frontend/public/resume/</code>{' '}
              folder.
            </p>
            <p className="text-sm text-gray-500">
              Supported formats: PDF files named resume.pdf, Resume.pdf, or RESUME.pdf
            </p>
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow-lg overflow-hidden">
            <iframe
              src={resumeFile}
              className="w-full h-screen"
              title="Resume"
            />
          </div>
        )}
      </div>
    </div>
  )
}
