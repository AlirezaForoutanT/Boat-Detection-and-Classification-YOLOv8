import Papa from 'papaparse';

/**
 * Extract CSV data to a JSON
 * @param {Array} csv Csv data array
 * @returns {Promise<Array>} Promise resolving to an array of JSON data
 */
const getCsvToJsonData = (csv) => {
  return new Promise((resolve, reject) => {
    try{
      let jsonData = [];

      // For each csv file we construct a unique object
      csv.forEach((csvFile) => {
        const csvString = csvFile.buffer.trim();
        
        Papa.parse(csvString, {
          header: true,
          complete: (result) => {
            jsonData = jsonData.concat(result.data);
          },
          error: (err) => {
            reject(new Error(`Error parsing CSV with papaparse: ${err}`));
          },
        });
      });

      // Sort the array with ascending date
      jsonData.sort((a, b) => {
        const dateA = new Date(a.date);
        const dateB = new Date(b.date);
        return dateA - dateB;
      });
      
      resolve(jsonData);
    } catch(err){
      reject(new Error(`Error parsing CSV with papaparse: ${err}`));
    }
  });
};

export default getCsvToJsonData;
